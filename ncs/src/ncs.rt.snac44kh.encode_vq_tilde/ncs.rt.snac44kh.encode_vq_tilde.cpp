// -----------------------------------------------------------------------------
// ncs.rt.snac44kh.encode_vq~  —  fused encode~ + vq
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

// Fuses ncs.rt.snac44kh.encode~ and ncs.rt.snac44kh.vq into ONE object
// running on ONE worker thread. Identical network to snac32kh (only
// sampling rate and weights differ) -- see
// ncs.rt.snac24kh.encode_vq_tilde.cpp for the full fusion rationale;
// this file only documents what's rate-specific. NOTE: encode_vq~'s own
// output (codes) is a small plain list, never chunked, so the
// kMaxSendsPerTick relaxed pacing used by standalone 44kh encode~/
// embedcodes for their large chunked embeddings outlets doesn't apply
// here -- same as standalone 44kh vq itself.
static constexpr int64_t kHopLength    = 384;
static constexpr int64_t kBlockFrames  = 32;                         // frames per block (== attn_window_size)
static constexpr int64_t kBlockSize    = kHopLength * kBlockFrames;  // 12288 samples
static constexpr int64_t kContextSize  = kBlockSize;                 // one block of trailing context
static constexpr int64_t kLatentDim    = 1024;
static constexpr int kNumLevels        = 4;

static constexpr double kModelSampleRate = 44100.0;

// Granularity (in HOST-rate samples) at which incoming audio is handed
// to the streaming resampler -- independent of Max's DSP vector size, so
// behavior doesn't change if the user changes their signal vector size.
static constexpr size_t kHostFlushSamples = 256;

// Real-time duration one block of NEW audio represents -- the divisor
// for @monitor_rtf's real-time factor (see PerformanceMonitorScope in
// shared_external_helpers.h), used by BOTH stages' timing outlets since
// both process the same block on the same real-time cadence.
static constexpr double kBlockDurationMs = static_cast<double>(kBlockSize) / kModelSampleRate * 1000.0;

// Carries a completed block from the audio thread to the worker thread.
// resample_ms is the time spent in resampler_.process() (operator(),
// audio thread) across every call that contributed to THIS block --
// accumulated separately because it happens outside process() (worker
// thread), but still has to count toward the encode stage's RTF/ms
// figure, same as encode_audio_segment.onnx's own Run() time.
struct EncodeJob {
    std::vector<float> combined;
    double resample_ms{0.0};
};

class NcsRtSnac44khEncodeVq : public object<NcsRtSnac44khEncodeVq>
    , public sample_operator<1, 0>
{
public:
    MIN_DESCRIPTION     {"Runs the SNAC 44kHz encoder and quantizer continuously on a live audio signal, fused into one object (no inter-object message relay between encode and vq)."};
    MIN_TAGS            {"snac, onnx, audio"};
    MIN_AUTHOR          {"Behzad Haki"};
    inlet<> signal_in{ this, "(signal) audio input" };
    outlet<> level0_out{ this, "(list) codes... -- codebook level 0 (coarsest, stride 8)" };
    outlet<> level1_out{ this, "(list) codes... -- codebook level 1 (stride 4)" };
    outlet<> level2_out{ this, "(list) codes... -- codebook level 2 (stride 2)" };
    outlet<> level3_out{ this, "(list) codes... -- codebook level 3 (finest, stride 1)" };
    // Declarative thread-safe outlet (see min-api's GuideToThreading,
    // "High-Level Outlet Threading Specification"): process() runs on
    // worker_thread_, not the main thread, so this defers the send to
    // the scheduler thread automatically. ONE outlet reporting the
    // TOTAL cost of both fused stages combined -- with everything now
    // running in one worker-thread call, there's no separate "encode
    // alone" or "vq alone" figure a user could act on differently, so a
    // single aggregate matches what's actually happening under the hood.
    outlet<thread_check::scheduler, thread_action::fifo> timing_out{
        this, "(float) real-time factor or process time in ms, per @monitor_rtf, emitted on every block" };

    attribute<bool> monitor_rtf{ this, "monitor_rtf", true,
        description{"Always emits the combined encode+vq per-block cost out timing_out. On (default): emitted as a real-time factor (elapsed/block-duration; >=1.0 means the fused stage can't keep up). Off: emitted as raw milliseconds."} };

    // =====================================================================
    // CONSTRUCTOR / DESTRUCTOR
    // =====================================================================
    NcsRtSnac44khEncodeVq()
        : output_timer_{this, MIN_FUNCTION { flush_output(); output_timer_.delay(10); return {}; }}
    {
        context_.assign(static_cast<size_t>(kContextSize), 0.0f);
        start_worker();
        start_output_timer();
        // Auto-load both models shipped alongside this external -- the
        // SAME encode_audio_segment.onnx / quantize_encodings.onnx the
        // separate modules use. `load_encode <path>` / `load_vq <path>`
        // still work afterwards to point at different models.
        default_model_path_encode_ = BundleResourceLoader::get_resource_path(
            "models/snac_onnx_exports/44khz/encode_audio_segment.onnx");
        default_model_path_vq_ = BundleResourceLoader::get_resource_path(
            "models/snac_onnx_exports/44khz/quantize_encodings.onnx");
        load_queue_encode_.enqueue(default_model_path_encode_);
        load_queue_vq_.enqueue(default_model_path_vq_);
    }

    ~NcsRtSnac44khEncodeVq() {
        stop_worker();
        stop_output_timer();
    }

    // Normally runs on the AUDIO THREAD only, and only ever from one
    // thread at a time -- but Max recompiles the DSP chain (and
    // re-registers this same perform callback) on every DSP on/off
    // toggle without destroying/recreating this object, and a toggle can
    // land in a narrow window where the previous chain's audio callback
    // hasn't fully quiesced before the new one starts. audio_mutex_
    // guards against that: if operator() is ever (re-)entered
    // concurrently, one call simply waits a few instructions for the
    // other instead of two threads mutating raw_accum_/model_pending_/
    // context_ at once -- which, left unguarded, is exactly the kind of
    // race that corrupts a std::vector's heap-allocated buffer. Never
    // calls into ONNX, min-api's error()/post(), or anything else that
    // can block/allocate unpredictably.
    sample operator()(sample in) {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        double sr = samplerate();
        if (reset_requested_.exchange(false) || sr != last_sr_seen_) {
            resampler_.reset_state();
            raw_accum_.clear();
            model_pending_.clear();
            context_.assign(static_cast<size_t>(kContextSize), 0.0f);
            block_resample_ms_ = 0.0;
            last_sr_seen_ = sr;
        }

        raw_accum_.push_back(static_cast<float>(in));
        if (raw_accum_.size() >= kHostFlushSamples) {
            double ratio = (sr > 0.0) ? (kModelSampleRate / sr) : 1.0;
            auto resample_t0 = std::chrono::steady_clock::now();
            auto resampled = resampler_.process(raw_accum_, ratio);
            block_resample_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - resample_t0).count();
            raw_accum_.clear();
            model_pending_.insert(model_pending_.end(), resampled.begin(), resampled.end());
        }

        while (model_pending_.size() >= static_cast<size_t>(kBlockSize)) {
            std::vector<float> new_block(model_pending_.begin(), model_pending_.begin() + kBlockSize);
            model_pending_.erase(model_pending_.begin(), model_pending_.begin() + kBlockSize);

            std::vector<float> combined;
            combined.reserve(static_cast<size_t>(kContextSize + kBlockSize));
            combined.insert(combined.end(), context_.begin(), context_.end());
            combined.insert(combined.end(), new_block.begin(), new_block.end());

            // Strict FIFO -- every block in this stream matters, none can
            // be dropped or collapsed (see ncs.rt.snac44kh.vq's
            // class-level comment).
            input_queue_.enqueue({std::move(combined), block_resample_ms_});
            block_resample_ms_ = 0.0;
            context_ = std::move(new_block);
        }
        return 0.0;
    }

    // Clears the streaming context (trailing audio history, resampler
    // filter state, and any not-yet-processed blocks) -- there's no real
    // conv state to reset (SNAC's ONNX graphs aren't stateful), so this
    // just re-primes the windowed context-trim bookkeeping for a fresh
    // stream. vq has no state of its own to reset (pointwise, no
    // temporal receptive field). Safe to call at any time:
    // reset_requested_ is only ever consumed by the audio thread itself,
    // so there's no race with the buffers it guards (all of which are
    // audio-thread-only).
    message<> reset_msg{this, "reset", "Clear streaming context and any queued (not-yet-encoded) blocks",
        MIN_FUNCTION {
            reset_requested_.store(true);
            input_queue_.clear();
            return {};
        }};

    message<> load_encode{this, "load_encode", "Load an ONNX model (.onnx) for the encode stage",
        MIN_FUNCTION {
            if (args.size() < 1) {
                log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: load_encode requires a file path"});
                return {};
            }
            load_queue_encode_.enqueue((std::string)args[0]);
            return {};
        }};

    message<> load_vq{this, "load_vq", "Load an ONNX model (.onnx) for the vq stage",
        MIN_FUNCTION {
            if (args.size() < 1) {
                log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: load_vq requires a file path"});
                return {};
            }
            load_queue_vq_.enqueue((std::string)args[0]);
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
    tsqueue<std::pair<int, atoms>> output_queue_; // outlet_id (0-3), codes...
    tsqueue<std::string> load_queue_encode_;
    tsqueue<std::string> load_queue_vq_;
    tsqueue<std::pair<bool,std::string>> log_queue_;

    // Audio-thread-only state (touched exclusively inside operator(),
    // under audio_mutex_ -- see operator()'s comment for why the mutex
    // exists despite this normally being single-threaded).
    std::mutex audio_mutex_;
    ncs_resample::StreamingResampler resampler_;
    std::vector<float> raw_accum_;      // host-rate samples awaiting resample
    std::vector<float> model_pending_;  // resampled (44.1kHz) samples awaiting a full block
    std::vector<float> context_;        // trailing kContextSize samples from the previous block
    double block_resample_ms_{0.0};     // accumulated resampler_.process() time for the block in progress
    double last_sr_seen_{-1.0};
    std::atomic<bool> reset_requested_{false};

    timer<> output_timer_;

    // Shared between both ONNX sessions -- just a thin wrapper around
    // the default ORT allocator, not model-specific state.
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
        if (worker_thread_.joinable()) {
            if (worker_thread_.get_id() != std::this_thread::get_id())
                worker_thread_.join();
            else
                worker_thread_.detach();
        }
        worker_running_ = false;
    }

    // Strict FIFO, no collapse-to-latest -- see operator()'s comment.
    // ONE worker thread runs BOTH stages sequentially per block (encode
    // Run() then vq Run()) -- vq's own compute is tiny compared to
    // encode's, so there's no meaningful throughput benefit to
    // pipelining them across two separate threads, only added
    // complexity.
    void worker_loop() {
        while (!stop_) {
            std::string path;
            if (load_queue_encode_.try_dequeue(path))
                load_model_encode(path);
            if (load_queue_vq_.try_dequeue(path))
                load_model_vq(path);
            EncodeJob item;
            if (!input_queue_.wait_dequeue(item, 100))
                continue;
            process(item);
        }
    }

    // Runs the full [context(kContextSize) + new(kBlockSize)] window
    // through encode_audio_segment.onnx, keeps only the LAST kBlockFrames
    // time-steps (the frames whose receptive field is anchored in real
    // context audio), then feeds that trimmed z DIRECTLY into
    // quantize_encodings.onnx -- no messageix/tensor serialization for
    // this handoff, it's a local std::vector passed straight from one
    // Run() to the next inside the same worker-thread call.
    void process(const EncodeJob& job) {
        const std::vector<float>& combined = job.combined;
        // Wraps BOTH stages -- one aggregate figure for the whole fused
        // block, not a per-stage breakdown (see timing_out's comment).
        PerformanceMonitorScope<decltype(timing_out)> perf_scope(
            bool(monitor_rtf), timing_out, kBlockDurationMs, job.resample_ms);

        if (!model_loaded_encode_) {
            log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: no encode model loaded"});
            return;
        }
        auto z = run_onnx_encode(combined);
        if (z.empty()) {
            log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: encode inference failed"});
            return;
        }
        int64_t T_total = static_cast<int64_t>(z.size()) / kLatentDim;
        if (T_total < kBlockFrames) {
            log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: unexpected encode output length"});
            return;
        }
        std::vector<float> trimmed(static_cast<size_t>(kLatentDim * kBlockFrames));
        for (int64_t c = 0; c < kLatentDim; ++c) {
            const float* src = &z[static_cast<size_t>(c * T_total + (T_total - kBlockFrames))];
            std::copy(src, src + kBlockFrames, trimmed.begin() + static_cast<size_t>(c * kBlockFrames));
        }

        if (!model_loaded_vq_) {
            log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: no vq model loaded"});
            return;
        }
        auto result = run_onnx_vq(trimmed);
        if (result.size() != kNumLevels) {
            log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: vq inference failed"});
            return;
        }
        // Same right-to-left send order as the standalone vq module,
        // for the same reason: a downstream embedcodes' cold inlets
        // land before its hot (level0) trigger.
        for (int lvl = kNumLevels - 1; lvl >= 0; --lvl) {
            atoms msg;
            msg.reserve(result[lvl].size());
            for (int v : result[lvl]) msg.push_back(v);
            output_queue_.enqueue({lvl, std::move(msg)});
        }
    }

    // Every crash seen so far in the buffer modules traced back to a call
    // to error() -- post() has been reliable throughout. Route everything
    // through post() with an "ERROR:" prefix instead. Exactly ONE outlet
    // per tick, same "never seen the exact trigger condition inside
    // Max's outlet_cache" precaution the buffer modules use.
    void flush_output() {
        std::pair<bool,std::string> log_msg;
        while (log_queue_.try_dequeue(log_msg)) {
            if (log_msg.first) c74::max::post("%s", ("ERROR: " + log_msg.second).c_str());
            else c74::max::post("%s", log_msg.second.c_str());
        }
        std::pair<int, atoms> item;
        if (output_queue_.try_dequeue(item)) {
            switch (item.first) {
                case 1: level1_out.send(item.second); break;
                case 2: level2_out.send(item.second); break;
                case 3: level3_out.send(item.second); break;
                default: level0_out.send(item.second); break;
            }
        }
    }

    void start_output_timer() {
        output_timer_.delay(10);
    }

    void stop_output_timer() {
        output_timer_.stop();
    }

    // min-api's error()/post() are only safe to call from the main thread —
    // calling them directly here (this runs on worker_thread_) can throw
    // from inside min-api itself, and an exception escaping a std::thread's
    // entry point is an instant abort(). So log_queue_ carries the message
    // to flush_output(), which the output_timer_ only ever runs on the main
    // thread, and it does the actual error()/post() call from there.
    void load_model_encode(const std::string& path) {
        try {
            session_encode_.reset();
            auto* env = ONNXManager::instance().get_env();
            if (!env) {
                log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: ONNX Runtime not available"});
                return;
            }
            session_options_encode_.SetIntraOpNumThreads(adaptive_intra_op_threads());
            session_options_encode_.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options_encode_.DisableCpuMemArena();
            session_options_encode_.DisableMemPattern();
            session_encode_ = std::make_unique<Ort::Session>(*env, path.c_str(),
                                                              session_options_encode_);
            ONNXManager::instance().release_env();

            auto input_type = session_encode_->GetInputTypeInfo(0);
            input_name_encode_ = session_encode_->GetInputNameAllocated(0, allocator_).get();
            input_dims_encode_ = input_type.GetTensorTypeAndShapeInfo().GetShape();

            auto output_type = session_encode_->GetOutputTypeInfo(0);
            output_name_encode_ = session_encode_->GetOutputNameAllocated(0, allocator_).get();
            output_dims_encode_ = output_type.GetTensorTypeAndShapeInfo().GetShape();

            model_loaded_encode_ = true;
        } catch (const std::exception& ex) {
            log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: failed to load encode model (" + path + ") — "
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
                log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: ONNX Runtime not available"});
                return;
            }
            session_options_vq_.SetIntraOpNumThreads(adaptive_intra_op_threads());
            session_options_vq_.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options_vq_.DisableCpuMemArena();
            session_options_vq_.DisableMemPattern();
            session_vq_ = std::make_unique<Ort::Session>(*env, path.c_str(),
                                                          session_options_vq_);
            ONNXManager::instance().release_env();

            auto input_type = session_vq_->GetInputTypeInfo(0);
            input_name_vq_ = session_vq_->GetInputNameAllocated(0, allocator_).get();
            input_dims_vq_ = input_type.GetTensorTypeAndShapeInfo().GetShape();

            size_t n_out = session_vq_->GetOutputCount();
            output_names_vq_.clear();
            for (size_t i = 0; i < n_out; ++i)
                output_names_vq_.push_back(session_vq_->GetOutputNameAllocated(i, allocator_).get());
            if (output_names_vq_.size() != kNumLevels) {
                log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: expected " + std::to_string(kNumLevels)
                                            + " codebook outputs, vq model has " + std::to_string(output_names_vq_.size())});
                model_loaded_vq_ = false;
                return;
            }

            model_loaded_vq_ = true;
        } catch (const std::exception& ex) {
            log_queue_.enqueue({true, "ncs.rt.snac44kh.encode_vq~: failed to load vq model (" + path + ") — "
                                        + std::string(ex.what()) + ". Models are expected at "
                                        + default_model_path_vq_ + "."});
            model_loaded_vq_ = false;
        }
    }

    std::vector<float> run_onnx_encode(const std::vector<float>& input) {
        if (!session_encode_ || !model_loaded_encode_) return {};
        try {
            std::vector<int64_t> shape = input_dims_encode_;
            if (!shape.empty() && shape[0] == -1) shape[0] = 1;
            if (!shape.empty() && shape.back() == -1) {
                int64_t known = 1;
                for (size_t i = 1; i + 1 < shape.size(); ++i) known *= shape[i];
                shape.back() = known > 0 ? static_cast<int64_t>(input.size()) / known
                                         : static_cast<int64_t>(input.size());
            }

            auto tensor = vector_to_tensor(input, shape, allocator_);
            const char* in_names[]  = {input_name_encode_.c_str()};
            const char* out_names[] = {output_name_encode_.c_str()};
            Ort::RunOptions opts;
            auto outputs = session_encode_->Run(opts, in_names, &tensor, 1,
                                                out_names, 1);
            if (outputs.size() > 0)
                return tensor_to_vector(outputs[0]);
        } catch (const std::exception&) {}
        return {};
    }

    // Runs quantize_encodings.onnx on flattened [1024 x T] embeddings and
    // returns each level's native-resolution codes as a separate int
    // vector (index 0 = coarsest/stride 8 ... index 3 = finest/stride 1).
    std::vector<std::vector<int>> run_onnx_vq(const std::vector<float>& z) {
        if (!session_vq_ || !model_loaded_vq_) return {};
        try {
            std::vector<int64_t> shape = input_dims_vq_;
            if (!shape.empty() && shape[0] == -1) shape[0] = 1;
            int64_t known = 1;
            for (size_t i = 1; i + 1 < shape.size(); ++i) known *= shape[i];
            int64_t T_b = known > 0 ? static_cast<int64_t>(z.size()) / known
                                     : static_cast<int64_t>(z.size());
            if (!shape.empty() && shape.back() == -1) shape.back() = T_b;

            auto tensor = vector_to_tensor(z, shape, allocator_);
            const char* in_names[] = {input_name_vq_.c_str()};
            std::vector<const char*> out_names;
            for (auto& n : output_names_vq_) out_names.push_back(n.c_str());

            Ort::RunOptions opts;
            auto outputs = session_vq_->Run(opts, in_names, &tensor, 1,
                                            out_names.data(), out_names.size());
            if (outputs.size() != kNumLevels) return {};

            std::vector<std::vector<int>> result(kNumLevels);
            for (int lvl = 0; lvl < kNumLevels; ++lvl) {
                auto raw = tensor_to_vector_i64(outputs[lvl]);
                result[lvl].assign(raw.begin(), raw.end());
            }
            return result;
        } catch (const std::exception&) {}
        return {};
    }
};

MIN_EXTERNAL(NcsRtSnac44khEncodeVq);
