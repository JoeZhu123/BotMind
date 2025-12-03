#pragma once

#include "backend.h"
#include <iostream>

// 实际项目中需要包含 CUDA 和 TensorRT 头文件
// #include <cuda_runtime.h>
// #include <NvInfer.h>

namespace botmind {

class NVBackend : public Backend {
public:
    NVBackend();
    ~NVBackend() override;

    bool Init() override;
    void* MemAlloc(size_t size) override;
    void MemFree(void* ptr) override;
    void MemCopy(void* dst, const void* src, size_t size, DeviceType dst_type, DeviceType src_type) override;
    bool Execute(const std::string& op_name, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
    void Synchronize() override;
    std::string GetDeviceName() const override;

private:
    // 模拟 CUDA stream
    void* stream_; 
    // 模拟 TensorRT context
    void* trt_context_;
};

} // namespace botmind

