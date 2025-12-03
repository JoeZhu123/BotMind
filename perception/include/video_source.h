#pragma once

#include <string>
#include <vector>
#include <memory>
#include "../../core/include/backend.h"

namespace botmind {

struct SensorFrame {
    std::vector<uint8_t> data; // RGB image or raw bytes
    int width;
    int height;
    int channels;
    int64_t timestamp;
    std::string sensor_id;
};

// 视频/传感器输入接口
class VideoSource {
public:
    virtual ~VideoSource() = default;

    // 打开设备 (e.g., "/dev/video0" or "rtsp://...")
    virtual bool Open(const std::string& uri) = 0;
    
    // 关闭设备
    virtual void Close() = 0;

    // 获取最新一帧
    virtual bool Capture(SensorFrame& frame) = 0;

    // 获取设备属性
    virtual int GetWidth() const = 0;
    virtual int GetHeight() const = 0;
};

// 简单的工厂模式
class VideoSourceFactory {
public:
    static std::shared_ptr<VideoSource> Create(const std::string& type);
};

} // namespace botmind

