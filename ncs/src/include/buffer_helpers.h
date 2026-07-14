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

inline std::vector<float> read_mono(c74::min::buffer_reference& ref) {
    using namespace c74::min;
    try {
        std::vector<float> out;
        if (!ref) return out;
        buffer_lock<false> b{ref};
        if (!b.valid()) return out;
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
