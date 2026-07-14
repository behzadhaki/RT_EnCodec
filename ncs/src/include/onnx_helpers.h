#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <mutex>
inline std::vector<float> tensor_to_vector(const Ort::Value&v){const float*p=v.GetTensorData<float>();size_t n=1;auto i=v.GetTensorTypeAndShapeInfo();for(auto d:i.GetShape())n*=static_cast<size_t>(d);return std::vector<float>(p,p+n);}
inline Ort::Value vector_to_tensor(const std::vector<float>&d,const std::vector<int64_t>&s,Ort::AllocatorWithDefaultOptions&a){size_t e=1;for(auto x:s)e*=static_cast<size_t>(x);Ort::Value t=Ort::Value::CreateTensor<float>(a,s.data(),s.size());std::memcpy(t.GetTensorMutableData<float>(),d.data(),e*sizeof(float));return t;}
inline Ort::Value vector_to_tensor_i64(const std::vector<int64_t>&d,const std::vector<int64_t>&s,Ort::AllocatorWithDefaultOptions&a){size_t e=1;for(auto x:s)e*=static_cast<size_t>(x);Ort::Value t=Ort::Value::CreateTensor<int64_t>(a,s.data(),s.size());std::memcpy(t.GetTensorMutableData<int64_t>(),d.data(),e*sizeof(int64_t));return t;}
inline std::vector<int64_t> tensor_to_vector_i64(const Ort::Value&v){const int64_t*p=v.GetTensorData<int64_t>();size_t n=1;auto i=v.GetTensorTypeAndShapeInfo();for(auto d:i.GetShape())n*=static_cast<size_t>(d);return std::vector<int64_t>(p,p+n);}

// Shared by every ncs.snac_24kh.* module: lets a new bang/message cancel a
// still-running Run() from an earlier one instead of waiting for it to
// finish, via Ort::RunOptions::SetTerminate() -- ONNX Runtime documents
// this as safe to call from a different thread than the one executing
// the Run() it targets. active_ is a non-owning pointer to a RunOptions
// that lives on the worker thread's stack for the exact duration of one
// run_onnx() call; only ever touched under mutex_, and always cleared
// (both the success and the exception path) before that stack frame
// unwinds, so cancel_active_run() (called from the main thread) can never
// see a dangling pointer.
class CancellableRun {
public:
    void cancel_active_run() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_) active_->SetTerminate();
    }
protected:
    void register_run(Ort::RunOptions* opts) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = opts;
    }
    void clear_run() {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = nullptr;
    }
private:
    std::mutex mutex_;
    Ort::RunOptions* active_{nullptr};
};
