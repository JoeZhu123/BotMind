import sys
import os
from typing import Any, Dict, Optional, Union, List
import numpy as np

# 尝试导入 botmind_core C++ 扩展
try:
    # 假设编译后的 .pyd/.so 在 python 目录下或系统路径中
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import botmind_core
except ImportError:
    print("[BotMind] Warning: C++ extension 'botmind_core' not found. Using pure Python mock mode.")
    botmind_core = None

class Engine:
    def __init__(self, backend_type: str = "nvidia"):
        self.backend_type = backend_type
        self.model = None
        print(f"[BotMind] Engine initialized with backend: {backend_type}")

        if botmind_core:
            if backend_type == "nvidia":
                self.cpp_backend = botmind_core.NVBackend()
            else:
                self.cpp_backend = botmind_core.CustomBackend()
            self.cpp_backend.init()
        else:
            self.cpp_backend = None

    def load_model_by_name(self, model_name: str):
        """
        根据模型名称从配置文件自动加载
        """
        # 延迟导入以避免循环依赖
        import json
        import importlib
        
        # 定位配置文件
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "configs", "model_zoo.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        model_info = config['model_zoo'].get(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found in model_zoo.json")
            
        # 动态导入 Adapter 类
        module_name, class_name = model_info['adapter_class'].rsplit('.', 1)
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        
        # 构造模型本地绝对路径
        local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", model_info['local_path']))
        
        print(f"[BotMind] Auto-loading {model_name} from {local_path}")
        self.load_adapter(adapter_cls, model_path=local_path)

    def load_adapter(self, adapter_class, *args, **kwargs):
        """
        加载 Python 层面的模型适配器 (如 X-VLA)
        """
        print(f"[BotMind] Loading adapter: {adapter_class.__name__}")
        self.model = adapter_class(self.backend_type, *args, **kwargs)

    def predict(self, *args, **kwargs):
        if self.model:
            return self.model.predict(*args, **kwargs)
        else:
            raise RuntimeError("No model loaded.")

