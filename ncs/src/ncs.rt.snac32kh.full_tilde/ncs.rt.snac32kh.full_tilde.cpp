// -----------------------------------------------------------------------------
// ncs.rt.snac32kh.full~  —  fused encode~ + vq + embedcodes + decode~
// -----------------------------------------------------------------------------

#ifdef _WIN32
#include <windows.h>
#endif

#include "c74_min.h"
#include "ext.h"
#include "../include/tsqueue.h"

#include "../include/shared_external_helpers.h"
#include "../include/onnx_helpers.h"
#include "../include/resample_helpers.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace c74::min;

// Fuses all four standalone SNAC 32kHz rt~ modules (encode~, vq,
// embedcodes, decode~) into ONE object with ONE worker thread running
// all four stages sequentially per block. Identical network to
// snac24kh (only sampling rate, block size, level count, and weights
// differ) -- see ncs.rt.snac24kh.full_tilde.cpp for the full fusion
// rationale; this file only documents what's rate-specific.
static constexpr int64_t kHopLength     = 384;
static constexpr int64_t kBlockFrames   = 32;                         // frames per block (also the encode-side context size, in frames)
static constexpr int64_t kBlockSize     = kHopLength * kBlockFrames;   // 12288 samples, encode-side context size in samples
static constexpr int64_t kContextFrames = kBlockFrames;               // decode-side: one block of trailing zq context
static constexpr int64_t kLatentDim     = 1024;
static constexpr int64_t kOlaSamples    = 256;                        // ~8ms @ 32kHz crossfade
static constexpr int kNumLevels         = 4;

static constexpr double kModelSampleRate = 32000.0;
static constexpr size_t kHostFlushSamples = 256;

// Divisor for @monitor_rtf's real-time factor -- one block of NEW audio,
// in ms, at the model's own sample rate. Shared by ALL FOUR stages'
// timing outlets, same as every other rt module this session.
static constexpr double kBlockDurationMs = static_cast<double>(kBlockSize) / kModelSampleRate * 1000.0;

// Carries a completed encode-side block AND the audio-thread resampling
// time it took to build it -- see encode~'s own EncodeJob comment for why
// resample time has to be threaded through separately (it runs outside
// the worker-thread process() call that owns the rest of this stage's
// PerformanceMonitorScope).
struct EncodeJob {
    std::vector<float> combined;
    double resample_ms{0.0};
};

// One vq -> embedcodes handoff: codes for all levels, plus each level's
// summing gain read at process() time. Local to this translation unit
// since the hop never crosses an object boundary here (unlike the
// standalone vq -> embedcodes hop, which is a Max message).
struct EmbedRequest {
    std::vector<int64_t> codes[kNumLevels];
    double scale[kNumLevels];
};

class NcsRtSnac32khFull : public object<NcsRtSnac32khFull>
    , public sample_operator<1, 1>
{
public:
    MIN_DESCRIPTION     {"Runs the full SNAC 32kHz encode -> vq -> embedcodes -> decode pipeline continuously on a signal input, fused into one object (no inter-object message relay between any of the four stages)."};
    MIN_TAGS            {"snac, onnx, audio"};
    MIN_AUTHOR          {"Behzad Haki"};

    inlet<>  signal_in{ this, "(signal) audio input" };
    outlet<> signal_out{ this, "(signal) audio output", "signal" };

    outlet<thread_check::scheduler, thread_action::fifo> underrun_out{
        this, "(bang) underrun state changed -- entering underrun (silence) or recovering from it" };

    // ONE outlet reporting the TOTAL cost of all four fused stages
    // combined -- see ncs.rt.snac32kh.encode_vq_tilde.cpp's timing_out
    // comment for why this isn't split into a per-stage breakdown. Only
    // decode~'s own outputs (signal_out, underrun_out) are exposed --
    // vq and embedcodes are internal stages here, not the last one in
    // the chain, so their raw codes/embeddings aren't surfaced as
    // separate outlets (unlike the standalone vq/embedcodes modules).
    outlet<thread_check::scheduler, thread_action::fifo> timing_out{
        this, "(float) real-time factor or process time in ms, per @monitor_rtf, emitted on every block" };

    attribute<bool> monitor_rtf{ this, "monitor_rtf", true,
        description{"Always emits the combined encode+vq+embedcodes+decode per-block cost out timing_out. On (default): emitted as a real-time factor (elapsed/block-duration; >=1.0 means the fused stage can't keep up). Off: emitted as raw milliseconds."} };

    attribute<int> prebuffer_blocks{ this, "prebuffer_blocks", 0,
        description{"Number of decoded blocks to accumulate before playback starts (or resumes after an underrun), absorbing transient processing slowdowns at the cost of added latency. 0 (default) disables prebuffering."} };

    attribute<number> level0_scale{ this, "level0_scale", 1.0,
        description{"Gain applied to codebook level 0 (coarsest) before summing into the decoded audio."} };
    attribute<number> level1_scale{ this, "level1_scale", 1.0,
        description{"Gain applied to codebook level 1 before summing into the decoded audio."} };
    attribute<number> level2_scale{ this, "level2_scale", 1.0,
        description{"Gain applied to codebook level 2 before summing into the decoded audio."} };
    attribute<number> level3_scale{ this, "level3_scale", 1.0,
        description{"Gain applied to codebook level 3 (finest) before summing into the decoded audio."} };

    // =====================================================================
    // CONSTRUCTOR / DESTRUCTOR
    // =====================================================================
    NcsRtSnac32khFull()
        : output_timer_{this, MIN_FUNCTION { flush_output(); output_timer_.delay(10); return {}; }}
    {
        context_.assign(static_cast<size_t>(kBlockSize), 0.0f);
        context_zq_.assign(static_cast<size_t>(kLatentDim * kContextFrames), 0.0f);
        start_worker();
        start_output_timer();

        default_model_path_encode_ = BundleResourceLoader::get_resource_path(
            "models/snac_onnx_exports/32khz/encode_audio_segment.onnx");
        default_model_path_vq_ = BundleResourceLoader::get_resource_path(
            "models/snac_onnx_exports/32khz/quantize_encodings.onnx");
        default_model_path_embedcodes_ = BundleResourceLoader::get_resource_path(
            "models/snac_onnx_exports/32khz/decode_codes.onnx");
        default_model_path_decode_ = BundleResourceLoader::get_resource_path(
            "models/snac_onnx_exports/32khz/decode_audio.onnx");
        load_queue_encode_.enqueue(default_model_path_encode_);
        load_queue_vq_.enqueue(default_model_path_vq_);
        load_queue_embedcodes_.enqueue(default_model_path_embedcodes_);
        load_queue_decode_.enqueue(default_model_path_decode_);
    }

    ~NcsRtSnac32khFull() {
        stop_worker();
        stop_output_timer();
    }

    // Audio thread. Two independent responsibilities, same split as the
    // standalone encode~/decode~ modules now fused into one callback:
    // (1) accumulate incoming samples into a model-rate block and hand
    // completed blocks to the worker thread; (2) serve already-decoded
    // samples for output, or silence during prebuffer/underrun. See
    // encode~'s and decode~'s own operator() comments for why each half
    // is structured the way it is (resampling into raw_accum_, the
    // windowed context-trim handoff, the prebuffer/underrun state
    // machine) -- unchanged here, just living in the same function body.
    // audio_mutex_ guards the same DSP-chain-recompile race described in
    // embedcodes_decode~'s operator() comment.
    sample operator()(sample in) {
        std::lock_guard<std::mutex> lock(audio_mutex_);

        // ---- encode-side: accumulate + resample + hand off ----
        double host_sr = samplerate();
        if (host_sr != last_encode_sr_seen_) {
            encode_resampler_.reset_state();
            last_encode_sr_seen_ = host_sr;
        }
        raw_accum_.push_back(static_cast<float>(in));
        if (raw_accum_.size() >= kHostFlushSamples) {
            auto t0 = std::chrono::steady_clock::now();
            double ratio = (host_sr > 0.0) ? (kModelSampleRate / host_sr) : 1.0;
            auto resampled = encode_resampler_.process(raw_accum_, ratio);
            raw_accum_.clear();
            auto t1 = std::chrono::steady_clock::now();
            double resample_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            model_pending_.insert(model_pending_.end(), resampled.begin(), resampled.end());
            block_resample_ms_ += resample_ms;
            while (model_pending_.size() >= static_cast<size_t>(kBlockSize)) {
                EncodeJob job;
                job.combined.assign(static_cast<size_t>(kBlockSize) * 2, 0.0f);
                std::copy(context_.begin(), context_.end(), job.combined.begin());
                std::copy(model_pending_.begin(), model_pending_.begin() + kBlockSize,
                          job.combined.begin() + kBlockSize);
                context_.assign(model_pending_.begin(), model_pending_.begin() + kBlockSize);
                job.resample_ms = block_resample_ms_;
                block_resample_ms_ = 0.0;
                model_pending_.erase(model_pending_.begin(), model_pending_.begin() + kBlockSize);
                input_queue_.enqueue(std::move(job));
            }
        }

        // ---- decode-side: playback state machine ----
        if (playback_reset_pending_.exchange(false)) {
            output_chunk_.clear();
            output_pos_ = 0;
            std::vector<float> discard;
            while (output_queue_.try_dequeue(discard)) {}
            prebuffering_ = true;
        }
        if (prebuffering_) {
            if (output_pos_ >= output_chunk_.size()
                && output_queue_.size() < static_cast<size_t>(std::max(0, static_cast<int>(prebuffer_blocks))))
                return 0.0;
            prebuffering_ = false;
        }
        if (output_pos_ >= output_chunk_.size()) {
            std::vector<float> next;
            if (output_queue_.try_dequeue(next)) {
                output_chunk_ = std::move(next);
                output_pos_ = 0;
            }
        }
        if (output_pos_ < output_chunk_.size()) {
            if (was_underrun_) {
                was_underrun_ = false;
                notify_underrun_transition();
            }
            return output_chunk_[output_pos_++];
        }
        if (!was_underrun_) {
            was_underrun_ = true;
            notify_underrun_transition();
            prebuffering_ = true;
        }
        return 0.0;
    }

    message<> reset_msg{this, "reset", "Clear streaming context, crossfade tail, and any queued/buffered audio",
        MIN_FUNCTION {
            context_.assign(static_cast<size_t>(kBlockSize), 0.0f);
            context_zq_.assign(static_cast<size_t>(kLatentDim * kContextFrames), 0.0f);
            model_pending_.clear();
            raw_accum_.clear();
            block_resample_ms_ = 0.0;
            input_queue_.clear();
            worker_reset_pending_.store(true);
            playback_reset_pending_.store(true);
            return {};
        }};

    message<> load_encode{this, "load_encode", "Load an ONNX model (.onnx) for the encode stage",
        MIN_FUNCTION {
            if (args.size() < 1) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: load_encode requires a file path"});
                return {};
            }
            load_queue_encode_.enqueue((std::string)args[0]);
            return {};
        }};

    message<> load_vq{this, "load_vq", "Load an ONNX model (.onnx) for the vq stage",
        MIN_FUNCTION {
            if (args.size() < 1) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: load_vq requires a file path"});
                return {};
            }
            load_queue_vq_.enqueue((std::string)args[0]);
            return {};
        }};

    message<> load_embedcodes{this, "load_embedcodes", "Load an ONNX model (.onnx) for the embedcodes stage",
        MIN_FUNCTION {
            if (args.size() < 1) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: load_embedcodes requires a file path"});
                return {};
            }
            load_queue_embedcodes_.enqueue((std::string)args[0]);
            return {};
        }};

    message<> load_decode{this, "load_decode", "Load an ONNX model (.onnx) for the decode stage",
        MIN_FUNCTION {
            if (args.size() < 1) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: load_decode requires a file path"});
                return {};
            }
            load_queue_decode_.enqueue((std::string)args[0]);
            return {};
        }};

// =============================================================================
// PRIVATE
// =============================================================================
private:
    std::thread worker_thread_;
    std::atomic<bool> worker_running_{false};
    std::atomic<bool> stop_{false};

    tsqueue<EncodeJob> input_queue_;
    tsqueue<std::vector<float>> output_queue_;
    tsqueue<std::string> load_queue_encode_, load_queue_vq_, load_queue_embedcodes_, load_queue_decode_;
    tsqueue<std::pair<bool,std::string>> log_queue_;

    // Audio-thread-only.
    std::mutex audio_mutex_;
    ncs_resample::StreamingResampler encode_resampler_;
    double last_encode_sr_seen_{-1.0};
    std::vector<float> raw_accum_;
    std::vector<float> model_pending_;
    std::vector<float> context_;           // encode-side trailing raw-audio context
    double block_resample_ms_{0.0};

    std::vector<float> output_chunk_;
    size_t output_pos_{0};
    bool was_underrun_{false};
    bool prebuffering_{true};
    std::chrono::steady_clock::time_point last_underrun_event_{};
    std::atomic<bool> playback_reset_pending_{false};

    // Worker-thread-only.
    std::vector<float> context_zq_;        // decode-side trailing zq context
    ncs_resample::StreamingResampler decode_resampler_;
    double last_decode_sr_seen_{-1.0};
    std::vector<float> held_tail_;
    std::atomic<bool> worker_reset_pending_{false};

    timer<> output_timer_;

    Ort::AllocatorWithDefaultOptions allocator_;

    // Encode stage (encode_audio_segment.onnx).
    bool model_loaded_encode_{false};
    std::string default_model_path_encode_;
    std::unique_ptr<Ort::Session> session_encode_;
    Ort::SessionOptions session_options_encode_;
    std::string input_name_encode_;
    std::string output_name_encode_;
    std::vector<int64_t> input_dims_encode_;
    std::vector<int64_t> output_dims_encode_;

    // VQ stage (quantize_encodings.onnx).
    bool model_loaded_vq_{false};
    std::string default_model_path_vq_;
    std::unique_ptr<Ort::Session> session_vq_;
    Ort::SessionOptions session_options_vq_;
    std::string input_name_vq_;
    std::vector<int64_t> input_dims_vq_;
    std::vector<std::string> output_names_vq_;

    // Embedcodes stage (decode_codes.onnx).
    bool model_loaded_embedcodes_{false};
    std::string default_model_path_embedcodes_;
    std::unique_ptr<Ort::Session> session_embedcodes_;
    Ort::SessionOptions session_options_embedcodes_;
    std::vector<std::string> input_names_embedcodes_;
    std::vector<std::string> output_names_embedcodes_;

    // Decode stage (decode_audio.onnx).
    bool model_loaded_decode_{false};
    std::string default_model_path_decode_;
    std::unique_ptr<Ort::Session> session_decode_;
    Ort::SessionOptions session_options_decode_;
    std::string input_name_decode_;
    std::string output_name_decode_;
    std::vector<int64_t> input_dims_decode_;
    std::vector<int64_t> output_dims_decode_;

    void start_worker() {
        if (worker_running_.exchange(true)) return;
        stop_ = false;
        worker_thread_ = std::thread([this] { worker_loop(); });
    }

    void stop_worker() {
        stop_ = true;
        input_queue_.shutdown();
        load_queue_encode_.shutdown();
        load_queue_vq_.shutdown();
        load_queue_embedcodes_.shutdown();
        load_queue_decode_.shutdown();
        if (worker_thread_.joinable()) {
            if (worker_thread_.get_id() != std::this_thread::get_id())
                worker_thread_.join();
            else
                worker_thread_.detach();
        }
        worker_running_ = false;
    }

    // Strict FIFO. ONE worker thread runs all FOUR stages sequentially
    // per block -- vq and embedcodes are both trivial compute next to
    // encode/decode, so there is no throughput benefit to splitting them
    // across separate threads; doing so would only add cross-thread
    // handoff overhead for no gain.
    void worker_loop() {
        while (!stop_) {
            std::string path;
            if (load_queue_encode_.try_dequeue(path)) load_model_encode(path);
            if (load_queue_vq_.try_dequeue(path)) load_model_vq(path);
            if (load_queue_embedcodes_.try_dequeue(path)) load_model_embedcodes(path);
            if (load_queue_decode_.try_dequeue(path)) load_model_decode(path);
            if (worker_reset_pending_.exchange(false)) {
                held_tail_.clear();
                decode_resampler_.reset_state();
                last_decode_sr_seen_ = -1.0;
            }
            EncodeJob job;
            if (!input_queue_.wait_dequeue(job, 100))
                continue;
            process(job);
        }
    }

    void process(const EncodeJob& job) {
        // Wraps ALL FOUR stages -- one aggregate figure for the whole
        // fused block, not a per-stage breakdown (see timing_out's
        // comment).
        PerformanceMonitorScope<decltype(timing_out)> perf_scope(
            bool(monitor_rtf), timing_out, kBlockDurationMs, job.resample_ms);

        // ---- ENCODE ----
        if (!model_loaded_encode_) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: no encode model loaded"});
            return;
        }
        auto z = run_onnx_encode(job.combined);
        if (z.empty()) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: encode inference failed"});
            return;
        }
        int64_t T_total = static_cast<int64_t>(z.size()) / kLatentDim;
        int64_t T_context = kBlockFrames; // one block of trailing context, in frames, mirrors encode~'s own trim
        if (T_total <= T_context) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: encode output shorter than expected context"});
            return;
        }
        int64_t T_new = T_total - T_context;
        std::vector<float> trimmed(static_cast<size_t>(kLatentDim * T_new));
        for (int64_t c = 0; c < kLatentDim; ++c) {
            const float* src = &z[static_cast<size_t>(c * T_total + T_context)];
            std::copy(src, src + T_new, trimmed.begin() + static_cast<size_t>(c * T_new));
        }

        // ---- VQ ----
        if (!model_loaded_vq_) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: no vq model loaded"});
            return;
        }
        auto codes = run_onnx_vq(trimmed, T_new);
        if (codes.size() != kNumLevels) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: vq inference failed"});
            return;
        }

        // ---- EMBEDCODES ----
        if (!model_loaded_embedcodes_) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: no embedcodes model loaded"});
            return;
        }
        EmbedRequest ereq;
        for (int lvl = 0; lvl < kNumLevels; ++lvl)
            ereq.codes[lvl].assign(codes[lvl].begin(), codes[lvl].end());
        ereq.scale[0] = double(level0_scale);
        ereq.scale[1] = double(level1_scale);
        ereq.scale[2] = double(level2_scale);
        ereq.scale[3] = double(level3_scale);
        auto sum = run_onnx_embedcodes(ereq);
        if (sum.empty()) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: embedcodes inference failed"});
            return;
        }
        int64_t T_block = static_cast<int64_t>(sum.size()) / kLatentDim;
        if (T_block <= 0 || T_block * kLatentDim != static_cast<int64_t>(sum.size())) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: summed embeddings size is not a multiple of 1024 channels"});
            return;
        }

        int64_t T_in = kContextFrames + T_block;
        std::vector<float> combined(static_cast<size_t>(kLatentDim * T_in));
        for (int64_t c = 0; c < kLatentDim; ++c) {
            float* dst = &combined[static_cast<size_t>(c * T_in)];
            std::copy(context_zq_.begin() + static_cast<size_t>(c * kContextFrames),
                      context_zq_.begin() + static_cast<size_t>(c * kContextFrames + kContextFrames),
                      dst);
            std::copy(sum.begin() + static_cast<size_t>(c * T_block),
                      sum.begin() + static_cast<size_t>(c * T_block + T_block),
                      dst + kContextFrames);
        }
        std::vector<float> new_context(static_cast<size_t>(kLatentDim * kContextFrames));
        for (int64_t c = 0; c < kLatentDim; ++c) {
            const float* src = &combined[static_cast<size_t>(c * T_in + (T_in - kContextFrames))];
            std::copy(src, src + kContextFrames, new_context.begin() + static_cast<size_t>(c * kContextFrames));
        }
        context_zq_ = std::move(new_context);

        // ---- DECODE ----
        if (!model_loaded_decode_) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: no decode model loaded"});
            return;
        }
        auto audio = run_onnx_decode(combined);
        if (audio.empty()) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: decode inference failed"});
            return;
        }
        int64_t total_samples = static_cast<int64_t>(audio.size());
        int64_t hop = T_block * kHopLength;
        int64_t boundary = total_samples - hop;
        if (boundary < 0 || hop <= 0) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: unexpected decode output length"});
            return;
        }
        int64_t ola = std::min<int64_t>(kOlaSamples, boundary);
        int64_t keep_start = boundary - ola;

        std::vector<float> kept(audio.begin() + keep_start, audio.end());
        std::vector<float> finalized(static_cast<size_t>(hop));
        for (int64_t i = 0; i < ola; ++i) {
            float t = (ola > 1) ? static_cast<float>(i) / static_cast<float>(ola - 1) : 1.0f;
            float old_v = (i < static_cast<int64_t>(held_tail_.size())) ? held_tail_[static_cast<size_t>(i)] : 0.0f;
            finalized[static_cast<size_t>(i)] = old_v * (1.0f - t) + kept[static_cast<size_t>(i)] * t;
        }
        if (hop > ola)
            std::copy(kept.begin() + ola, kept.begin() + hop, finalized.begin() + ola);
        held_tail_.assign(kept.begin() + hop, kept.end());

        double host_sr = samplerate();
        if (host_sr != last_decode_sr_seen_) {
            decode_resampler_.reset_state();
            last_decode_sr_seen_ = host_sr;
        }
        double ratio = (host_sr > 0.0) ? (host_sr / kModelSampleRate) : 1.0;
        auto resampled = decode_resampler_.process(finalized, ratio);
        output_queue_.enqueue(std::move(resampled));
    }

    void notify_underrun_transition() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_underrun_event_ < std::chrono::milliseconds(1000))
            return;
        last_underrun_event_ = now;
        underrun_out.send(k_sym_bang);
    }

    void flush_output() {
        std::pair<bool,std::string> log_msg;
        while (log_queue_.try_dequeue(log_msg)) {
            if (log_msg.first) c74::max::post("%s", ("ERROR: " + log_msg.second).c_str());
            else c74::max::post("%s", log_msg.second.c_str());
        }
    }

    void start_output_timer() { output_timer_.delay(10); }
    void stop_output_timer() { output_timer_.stop(); }

    void load_model_encode(const std::string& path) {
        try {
            session_encode_.reset();
            auto* env = ONNXManager::instance().get_env();
            if (!env) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: ONNX Runtime not available"});
                return;
            }
            session_options_encode_.SetIntraOpNumThreads(adaptive_intra_op_threads());
            session_options_encode_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options_encode_.DisableCpuMemArena();
            session_options_encode_.DisableMemPattern();
            session_encode_ = std::make_unique<Ort::Session>(*env, path.c_str(), session_options_encode_);
            ONNXManager::instance().release_env();

            input_name_encode_ = session_encode_->GetInputNameAllocated(0, allocator_).get();
            input_dims_encode_ = session_encode_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            output_name_encode_ = session_encode_->GetOutputNameAllocated(0, allocator_).get();
            output_dims_encode_ = session_encode_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

            model_loaded_encode_ = true;
        } catch (const std::exception& ex) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: failed to load encode model (" + path + ") — "
                                        + std::string(ex.what()) + ". Models are expected at "
                                        + default_model_path_encode_ + "."});
            model_loaded_encode_ = false;
        }
    }

    void load_model_vq(const std::string& path) {
        try {
            session_vq_.reset();
            auto* env = ONNXManager::instance().get_env();
            if (!env) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: ONNX Runtime not available"});
                return;
            }
            session_options_vq_.SetIntraOpNumThreads(adaptive_intra_op_threads());
            session_options_vq_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options_vq_.DisableCpuMemArena();
            session_options_vq_.DisableMemPattern();
            session_vq_ = std::make_unique<Ort::Session>(*env, path.c_str(), session_options_vq_);
            ONNXManager::instance().release_env();

            input_name_vq_ = session_vq_->GetInputNameAllocated(0, allocator_).get();
            input_dims_vq_ = session_vq_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

            size_t n_out = session_vq_->GetOutputCount();
            output_names_vq_.clear();
            for (size_t i = 0; i < n_out; ++i)
                output_names_vq_.push_back(session_vq_->GetOutputNameAllocated(i, allocator_).get());

            if (output_names_vq_.size() != kNumLevels) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: expected " + std::to_string(kNumLevels)
                                            + " vq outputs, model has " + std::to_string(output_names_vq_.size())});
                model_loaded_vq_ = false;
                return;
            }

            model_loaded_vq_ = true;
        } catch (const std::exception& ex) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: failed to load vq model (" + path + ") — "
                                        + std::string(ex.what()) + ". Models are expected at "
                                        + default_model_path_vq_ + "."});
            model_loaded_vq_ = false;
        }
    }

    void load_model_embedcodes(const std::string& path) {
        try {
            session_embedcodes_.reset();
            auto* env = ONNXManager::instance().get_env();
            if (!env) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: ONNX Runtime not available"});
                return;
            }
            session_options_embedcodes_.SetIntraOpNumThreads(adaptive_intra_op_threads());
            session_options_embedcodes_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options_embedcodes_.DisableCpuMemArena();
            session_options_embedcodes_.DisableMemPattern();
            session_embedcodes_ = std::make_unique<Ort::Session>(*env, path.c_str(), session_options_embedcodes_);
            ONNXManager::instance().release_env();

            size_t n_in = session_embedcodes_->GetInputCount();
            input_names_embedcodes_.clear();
            for (size_t i = 0; i < n_in; ++i)
                input_names_embedcodes_.push_back(session_embedcodes_->GetInputNameAllocated(i, allocator_).get());

            size_t n_out = session_embedcodes_->GetOutputCount();
            output_names_embedcodes_.clear();
            for (size_t i = 0; i < n_out; ++i)
                output_names_embedcodes_.push_back(session_embedcodes_->GetOutputNameAllocated(i, allocator_).get());

            if (input_names_embedcodes_.size() != kNumLevels || output_names_embedcodes_.size() != kNumLevels) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: expected " + std::to_string(kNumLevels)
                                            + " inputs/outputs, embedcodes model has " + std::to_string(input_names_embedcodes_.size())
                                            + " in / " + std::to_string(output_names_embedcodes_.size()) + " out"});
                model_loaded_embedcodes_ = false;
                return;
            }

            model_loaded_embedcodes_ = true;
        } catch (const std::exception& ex) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: failed to load embedcodes model (" + path + ") — "
                                        + std::string(ex.what()) + ". Models are expected at "
                                        + default_model_path_embedcodes_ + "."});
            model_loaded_embedcodes_ = false;
        }
    }

    void load_model_decode(const std::string& path) {
        try {
            session_decode_.reset();
            auto* env = ONNXManager::instance().get_env();
            if (!env) {
                log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: ONNX Runtime not available"});
                return;
            }
            session_options_decode_.SetIntraOpNumThreads(adaptive_intra_op_threads());
            session_options_decode_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options_decode_.DisableCpuMemArena();
            session_options_decode_.DisableMemPattern();
            session_decode_ = std::make_unique<Ort::Session>(*env, path.c_str(), session_options_decode_);
            ONNXManager::instance().release_env();

            input_name_decode_ = session_decode_->GetInputNameAllocated(0, allocator_).get();
            input_dims_decode_ = session_decode_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            output_name_decode_ = session_decode_->GetOutputNameAllocated(0, allocator_).get();
            output_dims_decode_ = session_decode_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

            model_loaded_decode_ = true;
        } catch (const std::exception& ex) {
            log_queue_.enqueue({true, "ncs.rt.snac32kh.full~: failed to load decode model (" + path + ") — "
                                        + std::string(ex.what()) + ". Models are expected at "
                                        + default_model_path_decode_ + "."});
            model_loaded_decode_ = false;
        }
    }

    std::vector<float> run_onnx_encode(const std::vector<float>& input) {
        if (!session_encode_ || !model_loaded_encode_) return {};
        try {
            std::vector<int64_t> shape = input_dims_encode_;
            if (!shape.empty() && shape[0] == -1) shape[0] = 1;
            if (!shape.empty() && shape.back() == -1) shape.back() = static_cast<int64_t>(input.size());

            auto tensor = vector_to_tensor(input, shape, allocator_);
            const char* in_names[]  = {input_name_encode_.c_str()};
            const char* out_names[] = {output_name_encode_.c_str()};
            Ort::RunOptions opts;
            auto outputs = session_encode_->Run(opts, in_names, &tensor, 1, out_names, 1);
            if (outputs.size() > 0)
                return tensor_to_vector(outputs[0]);
        } catch (const std::exception&) {}
        return {};
    }

    std::vector<std::vector<int>> run_onnx_vq(const std::vector<float>& z, int64_t T) {
        if (!session_vq_ || !model_loaded_vq_) return {};
        try {
            std::vector<int64_t> shape = {1, kLatentDim, T};
            auto tensor = vector_to_tensor(z, shape, allocator_);
            const char* in_names[] = {input_name_vq_.c_str()};
            std::vector<const char*> out_names;
            for (auto& n : output_names_vq_) out_names.push_back(n.c_str());

            Ort::RunOptions opts;
            auto outputs = session_vq_->Run(opts, in_names, &tensor, 1,
                                            out_names.data(), out_names.size());
            std::vector<std::vector<int>> result;
            result.reserve(outputs.size());
            for (auto& out : outputs) {
                auto floats = tensor_to_vector_i64(out);
                result.emplace_back(floats.begin(), floats.end());
            }
            return result;
        } catch (const std::exception&) {}
        return {};
    }

    // Returns the scaled sum of all levels, flattened [1024 x T] -- the
    // only thing decode~ (the last stage here) needs.
    std::vector<float> run_onnx_embedcodes(const EmbedRequest& req) {
        if (!session_embedcodes_ || !model_loaded_embedcodes_) return {};
        try {
            std::vector<Ort::Value> in_tensors;
            in_tensors.reserve(kNumLevels);
            std::vector<const char*> in_names;
            for (int lvl = 0; lvl < kNumLevels; ++lvl) {
                std::vector<int64_t> shape = {1, static_cast<int64_t>(req.codes[lvl].size())};
                in_tensors.push_back(vector_to_tensor_i64(req.codes[lvl], shape, allocator_));
                in_names.push_back(input_names_embedcodes_[lvl].c_str());
            }
            std::vector<const char*> out_names;
            for (auto& n : output_names_embedcodes_) out_names.push_back(n.c_str());

            Ort::RunOptions opts;
            auto outputs = session_embedcodes_->Run(opts, in_names.data(), in_tensors.data(), in_tensors.size(),
                                                     out_names.data(), out_names.size());
            if (outputs.size() != kNumLevels) return {};

            std::vector<float> zq[kNumLevels];
            for (int lvl = 0; lvl < kNumLevels; ++lvl)
                zq[lvl] = tensor_to_vector(outputs[lvl]);

            std::vector<float> sum(zq[0].size(), 0.0f);
            for (int lvl = 0; lvl < kNumLevels; ++lvl) {
                if (zq[lvl].size() != sum.size()) return {};
                float s = static_cast<float>(req.scale[lvl]);
                for (size_t i = 0; i < sum.size(); ++i)
                    sum[i] += zq[lvl][i] * s;
            }
            return sum;
        } catch (const std::exception&) {}
        return {};
    }

    std::vector<float> run_onnx_decode(const std::vector<float>& input) {
        if (!session_decode_ || !model_loaded_decode_) return {};
        try {
            std::vector<int64_t> shape = input_dims_decode_;
            if (!shape.empty() && shape[0] == -1) shape[0] = 1;
            if (!shape.empty() && shape.back() == -1) {
                int64_t known = 1;
                for (size_t i = 1; i + 1 < shape.size(); ++i) known *= shape[i];
                shape.back() = known > 0 ? static_cast<int64_t>(input.size()) / known
                                         : static_cast<int64_t>(input.size());
            }

            auto tensor = vector_to_tensor(input, shape, allocator_);
            const char* in_names[]  = {input_name_decode_.c_str()};
            const char* out_names[] = {output_name_decode_.c_str()};
            Ort::RunOptions opts;
            auto outputs = session_decode_->Run(opts, in_names, &tensor, 1, out_names, 1);
            if (outputs.size() > 0)
                return tensor_to_vector(outputs[0]);
        } catch (const std::exception&) {}
        return {};
    }

};

MIN_EXTERNAL(NcsRtSnac32khFull);
