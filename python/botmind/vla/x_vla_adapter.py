import sys
import os
import importlib
from typing import Union, List, Dict

# 帮助定位 third_party 目录
THIRD_PARTY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party"))
X_VLA_PATH = os.path.join(THIRD_PARTY_ROOT, "X-VLA")

class XVLAAdapter:
    def __init__(self, backend_type: str, model_path: str = "2toINF/X-VLA-WidowX"):
        self.backend_type = backend_type
        self.model_path = model_path
        self.model = None
        self.processor = None
        
        self._setup_env()
        self._load_model()

    def _setup_env(self):
        """将 X-VLA 添加到系统路径以便导入其模块"""
        if not os.path.exists(X_VLA_PATH):
            raise FileNotFoundError(
                f"X-VLA repository not found at {X_VLA_PATH}. "
                "Please clone it: `git clone https://github.com/2toinf/X-VLA third_party/X-VLA`"
            )
        
        if X_VLA_PATH not in sys.path:
            print(f"[XVLAAdapter] Adding {X_VLA_PATH} to sys.path")
            sys.path.insert(0, X_VLA_PATH)

    def _load_model(self):
        print(f"[XVLAAdapter] Loading X-VLA model from {self.model_path}...")
        
        try:
            # 使用 Transformers 加载
            from transformers import AutoModel, AutoProcessor
            
            # X-VLA 依赖 trust_remote_code=True 来加载自定义模型代码
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            
            if self.backend_type == "nvidia":
                print("[XVLAAdapter] Moving model to CUDA (NVIDIA Jetson/GPU)...")
                self.model = self.model.cuda()
            elif self.backend_type == "custom":
                print("[XVLAAdapter] Warning: Custom NPU backend for PyTorch not yet implemented. Keeping on CPU.")
                # 未来这里可以调用 torch.utils.cpp_extension 加载的自研芯片后端
                # self.model = self.model.to("privateuseone") 
            
            self.model.eval()
            print("[XVLAAdapter] Model loaded successfully.")
            
        except ImportError as e:
            print(f"[XVLAAdapter] Error: Failed to import necessary libraries. Ensure transformers and torch are installed. {e}")
            raise
        except Exception as e:
            print(f"[XVLAAdapter] Error loading model: {e}")
            raise

    def predict(self, image, instruction: str, proprio=None) -> Dict:
        """
        执行推理
        :param image: numpy array or PIL Image
        :param instruction: text instruction
        :param proprio: optional proprioception data
        """
        print(f"[XVLAAdapter] Predicting for instruction: '{instruction}'")
        
        # 模拟预处理，实际应参考 X-VLA 的 deploy.py 或 client.py
        # 这里只是一个示意性的 wrapper
        
        # inputs = self.processor(text=instruction, images=image, return_tensors="pt")
        # if self.backend_type == "nvidia":
        #     inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        
        # 假设直接调用 model.run 或者 forward
        # result = ...
        
        # 返回 Mock 数据以演示流程
        return {
            "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], # 6-DOF + Gripper
            "backend": self.backend_type
        }

