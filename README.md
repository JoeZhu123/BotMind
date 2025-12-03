# BotMind AI System Framework

BotMind 是一个专为具身智能（Embodied AI）设计的机器人大模型系统框架，支持 NVIDIA Jetson 平台及未来自研 AI 芯片。

## 目录结构
*   `core/`: 核心推理引擎与接口定义 (C++)
*   `backends/`: 硬件后端实现
*   `models/`: 模型实现
*   `python/`: Python SDK 与绑定
    *   `botmind/`: Python 包核心
        *   `perception/`: 传感器与视频处理
        *   `training/`: 数据预处理与训练工具
        *   `vla/`: 各种模型的 Adapter 实现 (X-VLA, OpenVLA, RT-2)
    *   `examples/`: 示例代码
*   `third_party/`: 第三方模型库集成
*   `perception/`: C++ 传感器采集底层实现
*   `training/`: 模型微调与预训练相关脚本

## 功能模块

### 1. 感知与采集 (Perception)
提供高性能的 C++ 视频/传感器采集接口，支持通过 Python 绑定调用。
*   **源码位置**: `perception/src`
*   **功能**: 封装摄像头 (V4L2/OpenCV) 及各类传感器读取，提供统一的 `SensorFrame`。

### 2. 数据预处理与训练 (Training)
提供视频数据的预处理流水线，用于 VLA 模型的微调或预训练。
*   **源码位置**: `python/botmind/training`
*   **功能**: 视频抽帧、尺寸归一化、数据集格式转换 (构建 `.pkl` 或 `.npy` 数据集)。

## 集成模型介绍 (Integrated Models)

### X-VLA: Soft-Prompted Transformer
BotMind 深度集成了 [X-VLA](https://thu-air-dream.github.io/X-VLA/)，这是一款在 **IROS 2025 AgiBot World Challenge** 中获得冠军的先进 VLA 模型。

*   **核心创新 (Soft Prompts)**: 引入具身特定的可学习嵌入（Embodiment-specific Soft Prompts），解决了多机器人平台异构数据的训练难题。
*   **架构优势**: 基于 Flow-matching 的 VLA 架构，采用标准 Transformer Encoder，仅需 0.9B 参数即可实现 SOTA 性能。
*   **通用能力**: 在 Simpler-WidowX 等仿真基准及真机测试（如叠衣物、灵巧手操作）中展现了极强的泛化能力和快速适应性。

### OpenVLA & Google RT-2
BotMind Model Zoo 同时也支持：
*   **OpenVLA (7B)**: 被社区广泛认为是开源版的 RT-2，性能强劲，已内置自动下载支持。
*   **Google RT-2**: 提供了标准 Adapter 接口 (`rt2_adapter.py`)，由于官方权重未公开，目前作为 Manual Import 占位符，方便有权限的用户手动接入。

## 快速开始

### 1. 环境准备
*   NVIDIA Jetson (Orin/Thor) 或 Linux PC
*   CUDA, TensorRT
*   Python 3.8+

### 2. 集成 X-VLA 模型
BotMind 支持直接调用开源的 [X-VLA](https://github.com/2toinf/X-VLA) 模型。请先将代码库克隆到 `third_party` 目录：

```bash
mkdir -p third_party
git clone https://github.com/2toinf/X-VLA third_party/X-VLA
```

### 3. 运行 X-VLA 示例
```bash
# 安装依赖
pip install torch transformers numpy

# 运行 Demo
python python/examples/demo_x_vla.py
```

### 4. 编译 C++ 核心 (可选，用于高性能生产环境)
```bash
mkdir build && cd build
cmake ..
make -j
```

## 模型管理 (Model Zoo)

BotMind 内置了 Model Zoo 管理机制，支持自动下载和统一调用。

### 常用命令
*   **列出可用模型**: `python python/botmind/download_models.py --list`
*   **下载指定模型**: `python python/botmind/download_models.py --model x-vla-widowx`
*   **一键下载所有**: `python python/botmind/download_models.py --all`

### 如何添加新模型
如果您需要集成新的模型（如 Google RT-2 或自定义模型），请遵循以下步骤：

1.  **注册模型信息**: 
    在 `models/configs/model_zoo.json` 中添加新的条目，定义模型来源（Hugging Face repo_id）和本地存储路径。
    ```json
    "my-new-model": {
        "source": "huggingface",
        "repo_id": "User/MyModel",
        "local_path": "models/huggingface/User/MyModel",
        "adapter_class": "botmind.vla.my_adapter.MyAdapter"
    }
    ```

2.  **编写适配器 (Adapter)**:
    在 `python/botmind/vla/` 下创建新的 Adapter 类（例如 `my_adapter.py`），实现统一的 `predict` 接口，负责处理该模型的预处理和推理逻辑。

3.  **调用**:
    ```python
    engine.load_model_by_name("my-new-model")
    ```

## 架构说明
BotMind 采用双层架构设计：
1.  **Python Layer (`botmind.engine`)**: 负责灵活的模型加载、数据预处理和快速原型验证。可以直接调用 PyTorch 模型（如 X-VLA）。
2.  **C++ Layer (`core`)**: 负责极致性能的算子执行。通过 `botmind_core` 绑定提供给 Python。
    *   对于 NVIDIA 平台，底层映射到 TensorRT/CUDA。
    *   对于自研芯片，通过 `CustomBackend` 映射到 NPU 指令集。
