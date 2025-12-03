#include "nv_backend.h"
#include <iostream>

namespace botmind {

NVBackend::NVBackend() : stream_(nullptr), trt_context_(nullptr) {}

NVBackend::~NVBackend() {
    // if (stream_) cudaStreamDestroy(stream_);
    // if (trt_context_) context->destroy();
    std::cout << "[NVBackend] Destroyed." << std::endl;
}

bool NVBackend::Init() {
    std::cout << "[NVBackend] Initializing on NVIDIA Jetson (Orin/Thor)..." << std::endl;
    // 实际代码: cudaStreamCreate(&stream_);
    // loadTensorRTModels();
    return true;
}

void* NVBackend::MemAlloc(size_t size) {
    // void* ptr;
    // cudaMalloc(&ptr, size);
    // return ptr;
    std::cout << "[NVBackend] Allocating " << size << " bytes on GPU." << std::endl;
    return malloc(size); // Mock
}

void NVBackend::MemFree(void* ptr) {
    // cudaFree(ptr);
    std::cout << "[NVBackend] Freeing GPU memory." << std::endl;
    free(ptr); // Mock
}

void NVBackend::MemCopy(void* dst, const void* src, size_t size, DeviceType dst_type, DeviceType src_type) {
    // cudaMemcpyAsync(dst, src, size, kind, stream_);
    std::cout << "[NVBackend] MemCopy Async." << std::endl;
}

bool NVBackend::Execute(const std::string& op_name, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    std::cout << "[NVBackend] Executing operator: " << op_name << " using TensorRT." << std::endl;
    // context->enqueueV2(bindings, stream_, nullptr);
    return true;
}

void NVBackend::Synchronize() {
    // cudaStreamSynchronize(stream_);
    std::cout << "[NVBackend] Synchronized." << std::endl;
}

std::string NVBackend::GetDeviceName() const {
    return "NVIDIA Jetson Orin/Thor";
}

} // namespace botmind

