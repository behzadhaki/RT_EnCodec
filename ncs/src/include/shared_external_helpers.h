#pragma once
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <libgen.h>
#include <sys/param.h>
#include <CoreFoundation/CFBundle.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Picks an intra-op thread count for an ONNX session that scales with
// the machine instead of a hardcoded value: too many threads on a
// low-core machine is oversubscription (hurts), too few on a high-core
// machine leaves compute on the table. Clamped to [1, 4] -- 4 was the
// value profiling justified for the rt~ pipeline's compute-heavy Run()
// calls on typical hardware; there's no evidence more helps these
// relatively small per-block tensors, so this only scales DOWN for
// weaker machines, not up for stronger ones. hardware_concurrency() can
// return 0 if it can't be determined, in which case this falls back to
// the original hardcoded 4.
inline int adaptive_intra_op_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 4;
    return static_cast<int>(std::max(1u, std::min(4u, hw)));
}

// RAII scope timer backing the user-facing @monitor_rtf attribute:
// ALWAYS sends a value out `outlet` on destruction (i.e. regardless of
// which return path the enclosing scope takes; and unconditionally --
// this is a live monitor, not a toggle). If as_rtf is true the value is
// the real-time factor (elapsed ms / block_duration_ms -- >=1.0 means
// this stage alone can no longer keep up with real-time); if false, the
// value is the raw elapsed time in ms.
//
// extra_ms folds in time spent OUTSIDE this scope that still counts
// toward the same block's total cost -- e.g. ncs.rt.snac_*.encode_tilde's
// resampling, which runs on the audio thread inside operator(), not in
// process() where this scope lives; pass 0.0 (the default) when nothing
// needs folding in.
//
// e.g. `PerformanceMonitorScope _perf(bool(monitor_rtf), timing_out, kBlockDurationMs);`
// as the first line of process(). outlet must be a thread-safe
// (thread_check::scheduler, thread_action::fifo) c74::min outlet, since
// this runs on a worker thread, not the main thread -- see
// ncs.rt.snac_44kh.decode_tilde.cpp's underrun_out for the same pattern.
template <typename Outlet>
class PerformanceMonitorScope {
public:
    PerformanceMonitorScope(bool as_rtf, Outlet& outlet, double block_duration_ms, double extra_ms = 0.0)
        : as_rtf_(as_rtf), outlet_(outlet), block_duration_ms_(block_duration_ms),
          extra_ms_(extra_ms), start_(std::chrono::steady_clock::now()) {}
    ~PerformanceMonitorScope() {
        double ms = extra_ms_ + std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_).count();
        double value = (as_rtf_ && block_duration_ms_ > 0.0) ? (ms / block_duration_ms_) : ms;
        // Sent as a 1-atom `atoms` list, not a bare double: outlet::send's
        // scalar-double overload constructs
        // handle_unsafe_outlet_send<check, thread_action::fifo, double>,
        // whose push() call in this vendored min-api tries to bind a
        // `double` to a `const atoms&` parameter and fails to compile.
        // The atoms overload's push() takes actual atoms, so it works.
        c74::min::atoms out{value};
        outlet_.send(out);
    }
private:
    bool as_rtf_;
    Outlet& outlet_;
    double block_duration_ms_;
    double extra_ms_;
    std::chrono::steady_clock::time_point start_;
};

class ONNXManager{
public:
static ONNXManager&instance(){static ONNXManager m;return m;}
Ort::Env*get_env(){std::lock_guard<std::mutex>l(m_);if(!env_){env_=new Ort::Env(ORT_LOGGING_LEVEL_WARNING,"max_onnx");env_->DisableTelemetryEvents();ref_count_=0;}ref_count_++;return env_;}
void release_env(){std::lock_guard<std::mutex>l(m_);if(ref_count_>0)ref_count_--;}
bool is_available(){std::lock_guard<std::mutex>l(m_);return env_!=nullptr;}
private:
ONNXManager()=default;~ONNXManager()=default;
ONNXManager(const ONNXManager&)=delete;
ONNXManager&operator=(const ONNXManager&)=delete;
Ort::Env*env_{nullptr};int ref_count_{0};std::mutex m_;
};
struct BundleResourceLoader{
static std::string get_resource_path(const std::string&r){auto b=base();return b.empty()?r:b+"/"+r;}
private:
static std::string base(){
#if defined(__APPLE__)
Dl_info i;if(dladdr((const void*)&base,&i)){std::string p(i.dli_fname);auto x=p.find("/externals/");if(x!=std::string::npos)return p.substr(0,x);x=p.find(".mxo/Contents/MacOS/");if(x!=std::string::npos)return p.substr(0,x+4)+"/Contents/Resources";}
CFArrayRef b=CFBundleGetAllBundles();if(b&&CFArrayGetCount(b)>0){CFBundleRef mb=(CFBundleRef)CFArrayGetValueAtIndex(b,0);CFURLRef u=CFBundleCopyResourcesDirectoryURL(mb);if(u){char buf[MAXPATHLEN];if(CFURLGetFileSystemRepresentation(u,true,(UInt8*)buf,MAXPATHLEN)){CFRelease(u);return std::string(buf);}CFRelease(u);}}
#elif defined(_WIN32)
HMODULE h=NULL;if(GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)&base,&h)){char p[MAX_PATH];GetModuleFileNameA(h,p,MAX_PATH);std::string s(p);auto x=s.find("\\externals\\");if(x!=std::string::npos)return s.substr(0,x);}
#endif
return"";
}
};
