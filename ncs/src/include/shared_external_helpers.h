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
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
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
