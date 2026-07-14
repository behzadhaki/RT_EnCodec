#pragma once
#include "c74_min.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

// Mono buffer~ read/write helpers built on c74::min::buffer_reference /
// buffer_lock<false>. Must only be called from the main thread (or the
// audio thread, with the other buffer_lock<> specialization) -- Max's
// buffer_edit_begin/buffer_edit_end are not documented as safe from an
// arbitrary spawned std::thread.
//
// Every entry point here is exception-safe (catches internally, returns a
// failure value instead of throwing): these run inside Max message
// callbacks invoked from Max's own C dispatch loop, and an exception
// escaping that boundary does not unwind cleanly -- it hits
// std::terminate() and takes the whole host process down.
namespace ncs_buffer {

// Guards against buffer_getframecount() ever returning a bogus size (e.g.
// a buffer~ that isn't fully initialized yet) turning into a
// multi-gigabyte vector::resize() attempt.
static constexpr size_t kMaxReasonableFrames = 100'000'000; // ~1000s @ 96kHz

// sr_out, if given, receives the buffer~'s OWN declared sample rate
// (t_buffer_info.b_sr) -- distinct from and independent of Max's live DSP
// driver rate. This pipeline is buffer-mode and never requires DSP/audio
// to be running, so the live driver rate (c74::max::sys_getsr()) can
// silently return a stale/default value; the buffer's own declared rate
// doesn't have that failure mode, since it's just data stored on the
// buffer~ itself.
inline std::vector<float> read_mono(c74::min::buffer_reference& ref, double* sr_out = nullptr) {
    using namespace c74::min;
    try {
        std::vector<float> out;
        if (!ref) return out;
        buffer_lock<false> b{ref};
        if (!b.valid()) return out;
        if (sr_out) *sr_out = b.samplerate();
        size_t frames = b.frame_count();
        if (frames == 0 || frames > kMaxReasonableFrames) return out;
        out.resize(frames);
        for (size_t i = 0; i < frames; ++i)
            out[i] = b.lookup(i, 0);
        return out;
    } catch (const std::exception&) {
        return {};
    }
}

// Same rationale as read_mono's sr_out -- queries the buffer~'s own
// declared sample rate without reading/writing its sample data, so
// ncs.snac_24kh.decode can pick its resample target before it has
// anything to write yet. Returns 0.0 if the buffer isn't set/valid.
inline double get_samplerate(c74::min::buffer_reference& ref) {
    using namespace c74::min;
    try {
        if (!ref) return 0.0;
        buffer_lock<false> b{ref};
        if (!b.valid()) return 0.0;
        return b.samplerate();
    } catch (const std::exception&) {
        return 0.0;
    }
}

// Resizes the buffer (if needed) and writes `data` into channel 0.
// Resize happens in its own lock scope -- the sample pointer captured by
// buffer_lock is invalidated by a resize, so the write must re-lock.
inline bool write_mono(c74::min::buffer_reference& ref, const std::vector<float>& data) {
    using namespace c74::min;
    if (data.empty() || data.size() > kMaxReasonableFrames) return false;
    try {
        {
            buffer_lock<false> b{ref};
            if (!b.valid()) return false;
            if (b.frame_count() != data.size())
                b.resize_in_samples(static_cast<int>(data.size()));
        }
        {
            buffer_lock<false> b{ref};
            if (!b.valid()) return false;
            for (size_t i = 0; i < data.size(); ++i)
                b.lookup(i, 0) = data[i];
            b.dirty();
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace ncs_buffer
