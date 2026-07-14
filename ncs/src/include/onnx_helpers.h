#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <cstring>
inline std::vector<float> tensor_to_vector(const Ort::Value&v){const float*p=v.GetTensorData<float>();size_t n=1;auto i=v.GetTensorTypeAndShapeInfo();for(auto d:i.GetShape())n*=static_cast<size_t>(d);return std::vector<float>(p,p+n);}
inline Ort::Value vector_to_tensor(const std::vector<float>&d,const std::vector<int64_t>&s,Ort::AllocatorWithDefaultOptions&a){size_t e=1;for(auto x:s)e*=static_cast<size_t>(x);Ort::Value t=Ort::Value::CreateTensor<float>(a,s.data(),s.size());std::memcpy(t.GetTensorMutableData<float>(),d.data(),e*sizeof(float));return t;}
inline Ort::Value vector_to_tensor_i64(const std::vector<int64_t>&d,const std::vector<int64_t>&s,Ort::AllocatorWithDefaultOptions&a){size_t e=1;for(auto x:s)e*=static_cast<size_t>(x);Ort::Value t=Ort::Value::CreateTensor<int64_t>(a,s.data(),s.size());std::memcpy(t.GetTensorMutableData<int64_t>(),d.data(),e*sizeof(int64_t));return t;}
inline std::vector<int64_t> tensor_to_vector_i64(const Ort::Value&v){const int64_t*p=v.GetTensorData<int64_t>();size_t n=1;auto i=v.GetTensorTypeAndShapeInfo();for(auto d:i.GetShape())n*=static_cast<size_t>(d);return std::vector<int64_t>(p,p+n);}
