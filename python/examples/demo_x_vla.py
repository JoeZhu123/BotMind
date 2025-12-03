import sys
import os
import numpy as np

# 将 python 目录添加到 path 以便导入 botmind 包
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from botmind.engine import Engine
from botmind.vla.x_vla_adapter import XVLAAdapter

def main():
    print("=== BotMind x X-VLA Demo ===")
    
    # 1. 初始化 BotMind 引擎 (指定后端)
    # 在 Jetson 上运行时，backend_type="nvidia"
    engine = Engine(backend_type="nvidia")
    
    # 2. 加载模型
    # 方式 A: 使用 Model Zoo 名称 (推荐)
    try:
        # 尝试加载 x-vla-widowx，如果未下载则需要先运行 python/botmind/download_models.py
        engine.load_model_by_name("x-vla-widowx")
    except Exception as e:
        print(f"自动加载失败: {e}")
        print("尝试手动加载默认路径...")
        
        # 方式 B: 手动指定 Adapter 和路径
        try:
            # engine.load_adapter(XVLAAdapter, model_path="2toINF/X-VLA-WidowX") # HuggingFace Remote
            pass
        except:
            pass
        return

    # 3. 准备输入数据 (Mock)
    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)
    instruction = "Pick up the red apple"
    
    # 4. 执行推理
    result = engine.predict(image=fake_image, instruction=instruction)
    
    print("\n推理结果 (Inference Result):")
    print(f"Action: {result['action']}")
    print(f"Backend: {result['backend']}")

if __name__ == "__main__":
    main()

