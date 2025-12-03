#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../models/vla_model.h"
#include "../backends/nvidia/nv_backend.h"
#include "../backends/custom/custom_backend.h"
#include "../perception/include/video_source.h"

namespace py = pybind11;
using namespace botmind;

PYBIND11_MODULE(botmind_core, m) {
    m.doc() = "BotMind AI Inference Framework Python Bindings";

    // --- Data Types ---
    py::class_<Action>(m, "Action")
        .def_readwrite("joint_positions", &Action::joint_positions)
        .def_readwrite("gripper_open", &Action::gripper_open);

    py::class_<SensorFrame>(m, "SensorFrame")
        .def_readwrite("data", &SensorFrame::data)
        .def_readwrite("width", &SensorFrame::width)
        .def_readwrite("height", &SensorFrame::height)
        .def_readwrite("timestamp", &SensorFrame::timestamp);

    // --- Backend ---
    py::class_<Backend, std::shared_ptr<Backend>>(m, "Backend");

    py::class_<NVBackend, Backend, std::shared_ptr<NVBackend>>(m, "NVBackend")
        .def(py::init<>())
        .def("init", &NVBackend::Init);

    py::class_<CustomChipBackend, Backend, std::shared_ptr<CustomChipBackend>>(m, "CustomBackend")
        .def(py::init<>())
        .def("init", &CustomChipBackend::Init);

    // --- Perception ---
    py::class_<VideoSource, std::shared_ptr<VideoSource>>(m, "VideoSource")
        .def("open", &VideoSource::Open)
        .def("close", &VideoSource::Close)
        .def("capture", &VideoSource::Capture)
        .def("get_width", &VideoSource::GetWidth)
        .def("get_height", &VideoSource::GetHeight);
    
    py::class_<VideoSourceFactory>(m, "VideoSourceFactory")
        .def_static("create", &VideoSourceFactory::Create);

    // --- Models ---
    py::class_<RobotVLA>(m, "RobotVLA")
        .def(py::init<std::shared_ptr<Backend>>())
        .def("load", &RobotVLA::Load)
        .def("predict", [](RobotVLA& self, const std::vector<uint8_t>& img, const std::string& text) {
            return self.Predict(img, text);
        });
}
