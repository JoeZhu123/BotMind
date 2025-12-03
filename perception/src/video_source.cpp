#include "video_source.h"
#include <iostream>
#include <chrono>
#include <cstring>

namespace botmind {

// 模拟 OpenCV VideoCapture
class OpenCVSource : public VideoSource {
public:
    bool Open(const std::string& uri) override {
        uri_ = uri;
        is_open_ = true;
        std::cout << "[Perception] Opened video source: " << uri << std::endl;
        return true;
    }

    void Close() override {
        is_open_ = false;
        std::cout << "[Perception] Closed video source." << std::endl;
    }

    bool Capture(SensorFrame& frame) override {
        if (!is_open_) return false;

        // Mock generating a frame
        frame.width = 640;
        frame.height = 480;
        frame.channels = 3;
        frame.sensor_id = uri_;
        
        // Allocate dummy data (simulating RGB)
        size_t size = frame.width * frame.height * frame.channels;
        frame.data.resize(size);
        
        // Fill with some pattern based on time
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        frame.timestamp = now;
        memset(frame.data.data(), (now % 255), size); 

        return true;
    }

    int GetWidth() const override { return 640; }
    int GetHeight() const override { return 480; }

private:
    std::string uri_;
    bool is_open_ = false;
};

std::shared_ptr<VideoSource> VideoSourceFactory::Create(const std::string& type) {
    if (type == "opencv" || type == "camera") {
        return std::make_shared<OpenCVSource>();
    }
    return nullptr;
}

} // namespace botmind

