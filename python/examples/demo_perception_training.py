import sys
import os
import time
import numpy as np

# 添加 python 根目录
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from botmind.engine import Engine
# 模拟 import，实际需要编译 botmind_core
try:
    import botmind_core
except ImportError:
    # Mock for demo
    class MockVideoSource:
        def open(self, uri): print(f"Open {uri}"); return True
        def close(self): print("Close")
        def capture(self, frame): 
            frame.data = [0]*640*480*3
            frame.width=640; frame.height=480
            return True
    class MockFactory:
        @staticmethod
        def create(t): return MockVideoSource()
    botmind_core = type('obj', (object,), {'VideoSourceFactory': MockFactory, 'SensorFrame': type('SF', (), {'data':[], 'width':0, 'height':0})})

from botmind.training.dataset_builder import DatasetBuilder

def run_collection_mode():
    print("=== 1. Data Acquisition Mode (C++ Perception) ===")
    # 1. 创建采集源
    factory = botmind_core.VideoSourceFactory()
    source = factory.create("camera")
    
    if source.open("/dev/video0"):
        print("Camera opened. Capturing 5 frames...")
        frame = botmind_core.SensorFrame()
        
        for i in range(5):
            if source.capture(frame):
                print(f"Frame {i}: {frame.width}x{frame.height}, Size={len(frame.data)}")
            time.sleep(0.1)
            
        source.close()

def run_training_prep_mode():
    print("\n=== 2. Training Preparation Mode (Python) ===")
    # 模拟创建一些虚拟视频文件
    os.makedirs("data/raw_videos", exist_ok=True)
    # (在真实场景中，这里会有实际的 mp4 文件)
    
    builder = DatasetBuilder(output_dir="data/training_set")
    # 扫描目录并处理
    # builder.build_from_videos("data/raw_videos")
    print("Dataset Builder initialized. Ready to process videos.")

if __name__ == "__main__":
    run_collection_mode()
    run_training_prep_mode()

