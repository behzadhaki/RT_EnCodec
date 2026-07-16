// -----------------------------------------------------------------------------
// ncs.rt.snac24kh.embedcodes_decode~  —  fused embedcodes + decode~
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

// Fuses ncs.rt.snac24kh.embedcodes and ncs.rt.snac24kh.decode~ into ONE
// object running on ONE worker thread. Functionally identical to
// patching embedcodes' sum_out into decode~'s embeddings_in -- same two
// ONNX graphs, same math -- but the handoff is a plain local variable
// instead of a chunked messageix/tensor Max message relayed through
// flush_output()'s 10ms timer (~10-40ms per hop after this session's
// transport tuning). Fusing removes that relay entirely for this hop.
// See ncs.rt.snac24kh.embedcodes.cpp and ncs.rt.snac24kh.decode_tilde.cpp
// for the full per-stage design rationale; this file only documents
// what's different about fusing them.
static constexpr int64_t kHopLength     = 512;
static constexpr int64_t kBlockFrames   = 4;                          // expected frames per block
static constexpr int64_t kContextFrames = kBlockFrames;               // one block of trailing zq context
static constexpr int64_t kLatentDim     = 768;
static constexpr int64_t kOlaSamples    = 256;                        // ~10.7ms @ 24kHz crossfade
static constexpr int kNumLevels         = 3;

static constexpr double kModelSampleRate = 24000.0;

// Real-time duration one block of NEW audio represents -- the divisor
// for @monitor_rtf's real-time factor (see PerformanceMonitorScope in
// shared_external_helpers.h). Used by BOTH stages' timing outlets.
// Unlike encode~'s fused counterpart, no separate resample-time folding
// is needed for the decode stage: decode_resampler_.process() already
// runs inside process() (see its call site below), so that scope's own
// elapsed time already covers Run() + OLA + resampling in full.
static constexpr double kBlockDurationMs = static_cast<double>(kBlockFrames * kHopLength) / kModelSampleRate * 1000.0;

// One incoming (hot-triggered) request: codes for all levels, plus each
// level's summing gain read at message-handling time.
struct EmbedRequest {
    std::vector<int64_t> codes[kNumLevels];
    double scale[kNumLevels];
};

class NcsRtSnac24khEmbedcodesDecode : public object<NcsRtSnac24khEmbedcodesDecode>
    , public sample_operator<0, 1>
{
public:
    MIN_DESCRIPTION     {"Runs the SNAC 24kHz codebook lookup and decoder continuously from a streamed per-level codes message, fused into one object (no inter-object message relay between embedcodes and decode)."};
    MIN_TAGS            {"snac, onnx, audio"};
    MIN_AUTHOR          {"Behzad Haki"};
    // Inlet 0 is hot (triggers processing using the most recently cached
    // level1/level2 values); inlets 1-2 are cold (cache only). This
    // matches ncs.rt.snac24kh.vq (or ncs.rt.snac24kh.encode_vq~), whose
    // outlets fire right-to-left (level2 first, level0 last), so a
    // direct patch cord hookup lands the cold values before the hot
    // trigger arrives.
    inlet<> level0_in{ this, "(list/load_embedcodes/load_decode) codes... from vq level0 -- hot, or model path" };
    inlet<> level1_in{ this, "(list) codes... from vq level1 -- cold" };
    inlet<> level2_in{ this, "(list) codes... from vq level2 -- cold" };

    outlet<> signal_out{ this, "(signal) audio output", "signal" };
    // Declarative thread-safe outlet (see min-api's GuideToThreading,
    // "High-Level Outlet Threading Specification"): calls made from the
    // audio thread are automatically deferred to the scheduler thread and
    // delivered in order, so operator() can call .send() directly with no
    // manual timer/queue plumbing of its own.
    outlet<thread_check::scheduler, thread_action::fifo> underrun_out{
        this, "(bang) underrun state changed -- entering underrun (silence) or recovering from it" };

    // Number of decoded blocks to accumulate in the output queue before
    // (re)starting playback. 0 (default) disables this -- audio plays as
    // soon as the first block is ready, identical to prior behavior.
    attribute<int> prebuffer_blocks{ this, "prebuffer_blocks", 0,
        description{"Number of decoded blocks to accumulate before playback starts (or resumes after an underrun), absorbing transient processing slowdowns at the cost of added latency. 0 (default) disables prebuffering."} };

    // Declarative thread-safe outlet, same pattern as underrun_out above:
    // process() runs on worker_thread_, not the main thread. ONE outlet
    // reporting the TOTAL cost of both fused stages combined -- see
    // ncs.rt.snac24kh.encode_vq_tilde.cpp's timing_out comment for why
    // this isn't split into a per-stage breakdown.
    outlet<thread_check::scheduler, thread_action::fifo> timing_out{
        this, "(float) real-time factor or process time in ms, per @monitor_rtf, emitted on every block" };

    attribute<bool> monitor_rtf{ this, "monitor_rtf", true,
        description{"Always emits the combined embedcodes+decode per-block cost out timing_out. On (default): emitted as a real-time factor (elapsed/block-duration; >=1.0 means the fused stage can't keep up). Off: emitted as raw milliseconds."} };

    attribute<number> level0_scale{ this, "level0_scale", 1.0,
        description{"Gain applied to codebook level 0 (coarsest) before summing into the decoded audio."} };
    attribute<number> level1_scale{ this, "level1_scale", 1.0,
        description{"Gain applied to codebook level 1 before summing into the decoded audio."} };
    attribute<number> level2_scale{ this, "level2_scale", 1.0,
        description{"Gain applied to codebook level 2 (finest) before summing into the decoded audio."} };

    // =====================================================================
    // CONSTRUCTOR / DESTRUCTOR
    // =====================================================================
    NcsRtSnac24khEmbedcodesDecode()
        : output_timer_{this, MIN_FUNCTION { flush_output(); output_timer_.delay(10); return {}; }}
    {
        context_zq_.assign(static_cast<size_t>(kLatentDim * kContextFrames), 0.0f);
        start_worker();
        start_output_timer();
        // Auto-load both models shipped alongside this external -- the
        // SAME decode_codes.onnx / decode_audio.onnx the separate
        // modules use. `load_embedcodes <path>` / `load_decode <path>`
        // still work afterwards to point at different models.
        default_model_path_embedcodes_ = BundleResourceLoader::get_resource_path(
            "models/snac_onnx_exports/24khz/decode_codes.onnx");
        default_model_path_decode_ = BundleResourceLoader::get_resource_path(
            "models/snac_onnx_exports/24khz/decode_audio.onnx");
        load_queue_embedcodes_.enqueue(default_model_path_embedcodes_);
        load_queue_decode_.enqueue(default_model_path_decode_);
    }

    ~NcsRtSnac24khEmbedcodesDecode() {
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
    // other instead of two threads mutating output_chunk_/output_pos_ at
    // once -- which, left unguarded, is exactly the kind of race that
    // corrupts a std::vector's heap-allocated buffer. Never blocks,
    // never calls into ONNX, or anything else that can lock/allocate
    // unpredictably (log_queue_.enqueue() is the same brief queue-push
    // the rest of this codebase already treats as audio-thread-safe).
    sample operator()() {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        if (playback_reset_pending_.exchange(false)) {
            output_chunk_.clear();
            output_pos_ = 0;
            std::vector<float> discard;
            while (output_queue_.try_dequeue(discard)) {}
            prebuffering_ = true;
        }
        // Pre-roll: while prebuffering, hold silence without consuming
        // from output_queue_ until it holds prebuffer_blocks worth of
        // decoded chunks (0 -- the default -- is always satisfied, so
        // this is a no-op unless the attribute is raised). output_queue_
        // is only touched here and just below via try_dequeue -- both
        // already-established audio-thread-safe brief-lock operations
        // (see the class comment above operator()).
        if (prebuffering_) {
            if (output_pos_ >= output_chunk_.size()
                && output_queue_.size() < static_cast<size_t>(std::max(0, static_cast<int>(prebuffer_blocks))))
                return 0.0;
            prebuffering_ = false;
        }
        // Serve samples from the current decoded chunk; when it runs
        // out, pull the next available chunk from the worker thread.
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
        // Underrun: nothing decoded yet -- output silence rather than
        // block waiting on the worker. Re-arms prebuffering (if enabled)
        // so playback resumes with a fresh reserve instead of immediately
        // racing the worker again one sample at a time.
        if (!was_underrun_) {
            was_underrun_ = true;
            notify_underrun_transition();
            prebuffering_ = true;
        }
        return 0.0;
    }

    // Parses vq's (or encode_vq~'s) format: a bare (headerless) codes
    // list per level. Errors go through log_queue_ (not error() inline)
    // -- this can fire very early during patch load, and calling error()
    // synchronously at that point has crashed Max's console/UI;
    // deferring it to the timer callback avoids that.
    message<> list_msg{this, "list", "Per-level codes block (inlet 0 hot / 1-2 cold)",
        MIN_FUNCTION {
            try {
                std::vector<int64_t> codes(args.size());
                for (size_t i = 0; i < args.size(); ++i)
                    codes[i] = (int64_t)(int)args[i];
                if (inlet == 1) {
                    cached_level1_ = std::move(codes);
                } else if (inlet == 2) {
                    cached_level2_ = std::move(codes);
                } else if (inlet == 0) {
                    // decode_codes.onnx structurally requires all 3 levels
                    // as input (not optional) -- an empty/missing level
                    // (e.g. vq's outlet 1 or 2 never connected here, so
                    // its cache stayed empty) turns into a zero-length
                    // tensor, which ONNX Runtime does not fail on
                    // gracefully. Refuse instead of risking that.
                    if (cached_level1_.empty() || cached_level2_.empty()) {
                        log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: level1 and/or level2 codes not yet received "
                              "— connect all three of vq's outlets to inlets 0-2"});
                        return {};
                    }
                    EmbedRequest req;
                    req.codes[0] = std::move(codes);
                    req.codes[1] = cached_level1_;
                    req.codes[2] = cached_level2_;
                    req.scale[0] = double(level0_scale);
                    req.scale[1] = double(level1_scale);
                    req.scale[2] = double(level2_scale);
                    // Strict FIFO -- every block in this stream matters;
                    // see ncs.rt.snac24kh.vq's class-level comment.
                    input_queue_.enqueue(std::move(req));
                }
            } catch (const std::exception& ex) {
                log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: list handling failed — " + std::string(ex.what())});
            }
            return {};
        }};

    // Clears the streaming context (trailing zq history, crossfade tail,
    // resampler filter state, currently-playing/queued audio, and any
    // not-yet-decoded blocks) -- there's no real conv state to reset
    // (SNAC's ONNX graphs aren't stateful), so this just re-primes the
    // windowed context-trim and OLA bookkeeping for a fresh stream.
    // context_zq_ is main-thread-owned (touched only here and in
    // list_msg, both on the main thread) so it's safe to clear directly;
    // the worker- and audio-thread-owned pieces go through their own
    // atomic flags, consumed by the thread that actually owns them.
    message<> reset_msg{this, "reset", "Clear streaming context, crossfade tail, and any queued/buffered audio",
        MIN_FUNCTION {
            context_zq_.assign(static_cast<size_t>(kLatentDim * kContextFrames), 0.0f);
            cached_level1_.clear();
            cached_level2_.clear();
            input_queue_.clear();
            worker_reset_pending_.store(true);
            playback_reset_pending_.store(true);
            return {};
        }};

    message<> load_embedcodes{this, "load_embedcodes", "Load an ONNX model (.onnx) for the embedcodes stage",
        MIN_FUNCTION {
            if (args.size() < 1) {
                log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: load_embedcodes requires a file path"});
                return {};
            }
            load_queue_embedcodes_.enqueue((std::string)args[0]);
            return {};
        }};

    message<> load_decode{this, "load_decode", "Load an ONNX model (.onnx) for the decode stage",
        MIN_FUNCTION {
            if (args.size() < 1) {
                log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: load_decode requires a file path"});
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

    // Only ever touched from message handlers, i.e. the main thread.
    std::vector<int64_t> cached_level1_;
    std::vector<int64_t> cached_level2_;
    std::vector<float> context_zq_;

    tsqueue<EmbedRequest> input_queue_;
    tsqueue<std::vector<float>> output_queue_;
    tsqueue<std::string> load_queue_embedcodes_;
    tsqueue<std::string> load_queue_decode_;
    tsqueue<std::pair<bool,std::string>> log_queue_;

    // Audio-thread-only (touched only inside operator(), under
    // audio_mutex_ -- see operator()'s comment for why the mutex exists
    // despite this normally being single-threaded).
    std::mutex audio_mutex_;
    std::vector<float> output_chunk_;
    size_t output_pos_{0};
    bool was_underrun_{false};
    bool prebuffering_{true};
    std::chrono::steady_clock::time_point last_underrun_event_{};
    std::atomic<bool> playback_reset_pending_{false};

    // Worker-thread-only (touched only inside process(), which only ever
    // runs sequentially on worker_thread_).
    ncs_resample::StreamingResampler decode_resampler_;
    double last_decode_sr_seen_{-1.0};
    std::vector<float> held_tail_; // last kOlaSamples of the previous block's own output, not yet emitted
    std::atomic<bool> worker_reset_pending_{false};

    timer<> output_timer_;

    // Shared between both ONNX sessions -- just a thin wrapper around
    // the default ORT allocator, not model-specific state.
    Ort::AllocatorWithDefaultOptions allocator_;

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

    // Strict FIFO, no collapse-to-latest -- see the class-level comment.
    // ONE worker thread runs BOTH stages sequentially per block
    // (embedcodes Run() then the decode context-trim+Run()+OLA+resample)
    // -- embedcodes' own compute is tiny (~5ms, measured earlier this
    // session) compared to decode's, so there's no meaningful throughput
    // benefit to pipelining them across two separate threads.
    void worker_loop() {
        while (!stop_) {
            std::string path;
            if (load_queue_embedcodes_.try_dequeue(path))
                load_model_embedcodes(path);
            if (load_queue_decode_.try_dequeue(path))
                load_model_decode(path);
            if (worker_reset_pending_.exchange(false)) {
                held_tail_.clear();
                decode_resampler_.reset_state();
                last_decode_sr_seen_ = -1.0;
            }
            EmbedRequest req;
            if (!input_queue_.wait_dequeue(req, 100))
                continue;
            process(req);
        }
    }

    // Runs decode_codes.onnx to get the summed embeddings, then feeds
    // them DIRECTLY into the decode stage's windowed context-trim +
    // decode_audio.onnx Run() + causal OLA crossfade + resample -- no
    // messageix/tensor serialization for this handoff, it's a local
    // std::vector passed straight from one Run() to the next inside the
    // same worker-thread call. Only decode~'s own outputs (signal_out,
    // underrun_out) are exposed -- embedcodes is an internal stage here,
    // not the last one in the chain, so its raw per-level embeddings
    // aren't surfaced as separate outlets (unlike the standalone
    // embedcodes module).
    void process(const EmbedRequest& req) {
        // Wraps BOTH stages -- one aggregate figure for the whole fused
        // block, not a per-stage breakdown (see timing_out's comment).
        PerformanceMonitorScope<decltype(timing_out)> perf_scope(
            bool(monitor_rtf), timing_out, kBlockDurationMs);

        if (!model_loaded_embedcodes_) {
            log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: no embedcodes model loaded"});
            return;
        }
        auto sum = run_onnx_embedcodes(req);
        if (sum.empty()) {
            log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: embedcodes inference failed"});
            return;
        }
        int64_t T_block = static_cast<int64_t>(sum.size()) / kLatentDim;
        if (T_block <= 0 || T_block * kLatentDim != static_cast<int64_t>(sum.size())) {
            log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: summed embeddings size is not a multiple of 768 channels"});
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

        if (!model_loaded_decode_) {
            log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: no decode model loaded"});
            return;
        }
        auto audio = run_onnx_decode(combined);
        if (audio.empty()) {
            log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: decode inference failed"});
            return;
        }
        int64_t total_samples = static_cast<int64_t>(audio.size());
        int64_t hop = T_block * kHopLength;
        int64_t boundary = total_samples - hop; // == kContextFrames*kHopLength in the normal case
        if (boundary < 0 || hop <= 0) {
            log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: unexpected decode output length"});
            return;
        }
        int64_t ola = std::min<int64_t>(kOlaSamples, boundary);
        int64_t keep_start = boundary - ola;

        std::vector<float> kept(audio.begin() + keep_start, audio.end()); // length hop+ola
        std::vector<float> finalized(static_cast<size_t>(hop));
        for (int64_t i = 0; i < ola; ++i) {
            float t = (ola > 1) ? static_cast<float>(i) / static_cast<float>(ola - 1) : 1.0f;
            float old_v = (i < static_cast<int64_t>(held_tail_.size())) ? held_tail_[static_cast<size_t>(i)] : 0.0f;
            finalized[static_cast<size_t>(i)] = old_v * (1.0f - t) + kept[static_cast<size_t>(i)] * t;
        }
        if (hop > ola)
            std::copy(kept.begin() + ola, kept.begin() + hop, finalized.begin() + ola);
        held_tail_.assign(kept.begin() + hop, kept.end()); // length ola, held for the NEXT call

        double host_sr = samplerate();
        if (host_sr != last_decode_sr_seen_) {
            decode_resampler_.reset_state();
            last_decode_sr_seen_ = host_sr;
        }
        double ratio = (host_sr > 0.0) ? (host_sr / kModelSampleRate) : 1.0;
        auto resampled = decode_resampler_.process(finalized, ratio);
        output_queue_.enqueue(std::move(resampled));
    }

    // Bangs underrun_out on both the entering-underrun and the
    // recovering-from-underrun transitions, in place of a console log.
    // Rate-limited to at most one bang per second regardless of how many
    // times the state actually flips within that window, since
    // underrun_out's own fifo defer-to-scheduler queue has no bound on
    // how many bangs it will accumulate before the next scheduler tick
    // delivers them. Called from the audio thread (under audio_mutex_,
    // like the rest of operator()); std::chrono is safe to use there,
    // and underrun_out.send() itself never blocks -- it just pushes onto
    // its own lock-free fifo (see the underrun_out declaration's comment).
    void notify_underrun_transition() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_underrun_event_ < std::chrono::milliseconds(1000))
            return;
        last_underrun_event_ = now;
        underrun_out.send(k_sym_bang);
    }

    // Every crash seen so far in the buffer modules traced back to a call
    // to error() -- post() has been reliable throughout. Route everything
    // through post() with an "ERROR:" prefix instead. Exactly ONE outlet
    // per tick, same precaution the buffer modules use.
    void flush_output() {
        std::pair<bool,std::string> log_msg;
        while (log_queue_.try_dequeue(log_msg)) {
            if (log_msg.first) c74::max::post("%s", ("ERROR: " + log_msg.second).c_str());
            else c74::max::post("%s", log_msg.second.c_str());
        }
    }

    void start_output_timer() {
        output_timer_.delay(10);
    }

    void stop_output_timer() {
        output_timer_.stop();
    }

    void load_model_embedcodes(const std::string& path) {
        try {
            session_embedcodes_.reset();
            auto* env = ONNXManager::instance().get_env();
            if (!env) {
                log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: ONNX Runtime not available"});
                return;
            }
            session_options_embedcodes_.SetIntraOpNumThreads(1);
            session_options_embedcodes_.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options_embedcodes_.DisableCpuMemArena();
            session_options_embedcodes_.DisableMemPattern();
            session_embedcodes_ = std::make_unique<Ort::Session>(*env, path.c_str(),
                                                                  session_options_embedcodes_);
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
                log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: expected " + std::to_string(kNumLevels)
                                            + " inputs/outputs, embedcodes model has " + std::to_string(input_names_embedcodes_.size())
                                            + " in / " + std::to_string(output_names_embedcodes_.size()) + " out"});
                model_loaded_embedcodes_ = false;
                return;
            }

            model_loaded_embedcodes_ = true;
        } catch (const std::exception& ex) {
            log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: failed to load embedcodes model (" + path + ") — "
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
                log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: ONNX Runtime not available"});
                return;
            }
            session_options_decode_.SetIntraOpNumThreads(1);
            session_options_decode_.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options_decode_.DisableCpuMemArena();
            session_options_decode_.DisableMemPattern();
            session_decode_ = std::make_unique<Ort::Session>(*env, path.c_str(),
                                                              session_options_decode_);
            ONNXManager::instance().release_env();

            auto input_type = session_decode_->GetInputTypeInfo(0);
            input_name_decode_ = session_decode_->GetInputNameAllocated(0, allocator_).get();
            input_dims_decode_ = input_type.GetTensorTypeAndShapeInfo().GetShape();

            auto output_type = session_decode_->GetOutputTypeInfo(0);
            output_name_decode_ = session_decode_->GetOutputNameAllocated(0, allocator_).get();
            output_dims_decode_ = output_type.GetTensorTypeAndShapeInfo().GetShape();

            model_loaded_decode_ = true;
        } catch (const std::exception& ex) {
            log_queue_.enqueue({true, "ncs.rt.snac24kh.embedcodes_decode~: failed to load decode model (" + path + ") — "
                                        + std::string(ex.what()) + ". Models are expected at "
                                        + default_model_path_decode_ + "."});
            model_loaded_decode_ = false;
        }
    }

    // Runs decode_codes.onnx on the three native-resolution per-level code
    // sequences and returns the scaled sum of all levels, flattened
    // [768 x T] -- the only thing decode~ (the last stage here) needs.
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
                // z_q is [B, 768, T] -- the flattened input covers every
                // dim, so the dynamic trailing dim is input.size() divided
                // by the product of the static middle dims (768), not
                // input.size() itself.
                int64_t known = 1;
                for (size_t i = 1; i + 1 < shape.size(); ++i) known *= shape[i];
                shape.back() = known > 0 ? static_cast<int64_t>(input.size()) / known
                                         : static_cast<int64_t>(input.size());
            }

            auto tensor = vector_to_tensor(input, shape, allocator_);
            const char* in_names[]  = {input_name_decode_.c_str()};
            const char* out_names[] = {output_name_decode_.c_str()};
            Ort::RunOptions opts;
            auto outputs = session_decode_->Run(opts, in_names, &tensor, 1,
                                                out_names, 1);
            if (outputs.size() > 0)
                return tensor_to_vector(outputs[0]);
        } catch (const std::exception&) {}
        return {};
    }
};

MIN_EXTERNAL(NcsRtSnac24khEmbedcodesDecode);
