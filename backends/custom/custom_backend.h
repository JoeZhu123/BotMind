#pragma once

#include "backend.h"

namespace botmind {

// 自研AI芯片后端示例
class CustomChipBackend : public Backend {
public:
    CustomChipBackend();
    ~CustomChipBackend() override;

    bool Init() override;
    void* MemAlloc(size_t size) override;
    void MemFree(void* ptr) override;
    void MemCopy(void* dst, const void* src, size_t size, DeviceType dst_type, DeviceType src_type) override;
    bool Execute(const std::string& op_name, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
    void Synchronize() override;
    std::string GetDeviceName() const override;
};

} // namespace botmind

