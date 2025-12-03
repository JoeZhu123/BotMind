#pragma once

#include <vector>
#include <string>
#include <memory>

namespace botmind {

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT8,
    INT32
};

enum class DeviceType {
    CPU,
    NVIDIA_GPU,
    CUSTOM_NPU
};

// 通用张量定义
struct Tensor {
    void* data;
    std::vector<int64_t> shape;
    DataType dtype;
    DeviceType device;
    
    size_t GetByteSize() const;
};

// 硬件后端抽象接口
// 所有具体的硬件支持（NV Orin/Thor, 自研芯片）都需要实现此接口
class Backend {
public:
    virtual ~Backend() = default;

    // 初始化设备环境
    virtual bool Init() = 0;

    // 内存管理
    virtual void* MemAlloc(size_t size) = 0;
    virtual void MemFree(void* ptr) = 0;
    virtual void MemCopy(void* dst, const void* src, size_t size, DeviceType dst_type, DeviceType src_type) = 0;

    // 执行算子或计算图
    // 这里简化为一个通用的Execute接口，实际可能需要更复杂的Graph或Operator定义
    virtual bool Execute(const std::string& op_name, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) = 0;

    // 设备同步
    virtual void Synchronize() = 0;

    // 获取设备信息
    virtual std::string GetDeviceName() const = 0;
};

} // namespace botmind

