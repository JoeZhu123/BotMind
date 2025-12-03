#pragma once

#include <string>
#include <vector>
#include <memory>
#include "../core/include/backend.h"

namespace botmind {

// 机器人控制指令
struct Action {
    std::vector<float> joint_positions;
    std::vector<float> joint_velocities;
    bool gripper_open;
};

// VLA模型基类
class VLAModel {
public:
    VLAModel(std::shared_ptr<Backend> backend) : backend_(backend) {}
    virtual ~VLAModel() = default;

    // 加载模型权重
    virtual bool Load(const std::string& model_path) = 0;

    // 端到端推理：图像+文本指令 -> 动作
    // img_data: 原始RGB数据
    // instruction: 文本指令 (e.g., "Pick up the red apple")
    virtual Action Predict(const std::vector<uint8_t>& img_data, const std::string& instruction) = 0;

protected:
    std::shared_ptr<Backend> backend_;
};

// 具体的VLA模型实现 (例如基于 OpenVLA 或 RT-2)
class RobotVLA : public VLAModel {
public:
    using VLAModel::VLAModel;

    bool Load(const std::string& model_path) override;
    Action Predict(const std::vector<uint8_t>& img_data, const std::string& instruction) override;

private:
    // 内部处理函数
    void EncodeImage(const std::vector<uint8_t>& img_data, Tensor& embedding);
    void EncodeText(const std::string& text, Tensor& embedding);
    Action DecodeAction(const Tensor& output_logits);
};

} // namespace botmind

