#pragma once
#include <samplerate.h>
#include <cmath>
#include <vector>

// One-shot (non-streaming) resampling for buffer-mode processing: each
// call is independent, no persistent filter state carried across calls
// (unlike real-time signal modules, which need SRC_STATE continuity
// across consecutive small blocks to avoid clicking at block boundaries).
// A whole buffer is read/processed/written in one shot here, so
// src_simple() is the right (simpler) libsamplerate entry point.
namespace ncs_resample {

inline std::vector<float> resample_simple(const std::vector<float>& in, double ratio) {
    if (in.empty() || ratio <= 0.0 || std::abs(ratio - 1.0) < 1e-9)
        return in;
    std::vector<float> out(static_cast<size_t>(static_cast<double>(in.size()) * ratio) + 16);
    SRC_DATA d{};
    d.data_in = in.data();
    d.input_frames = static_cast<long>(in.size());
    d.data_out = out.data();
    d.output_frames = static_cast<long>(out.size());
    d.src_ratio = ratio;
    int err = src_simple(&d, SRC_SINC_BEST_QUALITY, 1);
    if (err != 0) return in; // fall back to unresampled rather than lose the data
    out.resize(static_cast<size_t>(d.output_frames_gen));
    return out;
}

// Persistent-state resampler for the rt~ (streaming) modules: each call is
// one link in a continuous chain (unlike resample_simple's one-shot buffer
// calls), so the SRC_STATE filter history must survive across calls or
// every block boundary clicks. SRC_SINC_FASTEST (not BEST_QUALITY) -- this
// runs continuously on every rt~ block, not once per buffer-mode bang, so
// the cheaper converter trades a little quality for headroom.
class StreamingResampler {
public:
    StreamingResampler() = default;
    ~StreamingResampler() { reset_state(); }
    StreamingResampler(const StreamingResampler&) = delete;
    StreamingResampler& operator=(const StreamingResampler&) = delete;

    // Drops the filter history -- call whenever the input stream is no
    // longer contiguous with what came before (host samplerate changed,
    // or an explicit user-facing reset/clear).
    void reset_state() {
        if (state_) { src_delete(state_); state_ = nullptr; }
    }

    std::vector<float> process(const std::vector<float>& in, double ratio) {
        if (in.empty()) return {};
        if (ratio <= 0.0 || std::abs(ratio - 1.0) < 1e-9) return in;
        if (!state_) {
            int err = 0;
            state_ = src_new(SRC_SINC_FASTEST, 1, &err);
            if (!state_) return in; // fall back to unresampled rather than lose the data
        }
        std::vector<float> out(static_cast<size_t>(static_cast<double>(in.size()) * ratio) + 32);
        SRC_DATA d{};
        d.data_in = in.data();
        d.input_frames = static_cast<long>(in.size());
        d.data_out = out.data();
        d.output_frames = static_cast<long>(out.size());
        d.src_ratio = ratio;
        d.end_of_input = 0;
        int err = src_process(state_, &d);
        if (err != 0) return in;
        out.resize(static_cast<size_t>(d.output_frames_gen));
        return out;
    }

private:
    SRC_STATE* state_{nullptr};
};

} // namespace ncs_resample
