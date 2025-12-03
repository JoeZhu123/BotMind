import sys
import os
import torch
from typing import Dict

# 帮助定位
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class OpenVLAAdapter:
    def __init__(self, backend_type: str, model_path: str):
        self.backend_type = backend_type
        self.model_path = model_path
        self.model = None
        self.processor = None
        
        self._load_model()

    def _load_model(self):
        print(f"[OpenVLAAdapter] Loading OpenVLA from {self.model_path}...")
        
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            # OpenVLA 7B 通常需要 bfloat16 
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True
            )
            
            if self.backend_type == "nvidia":
                print("[OpenVLAAdapter] Moving model to CUDA...")
                self.model = self.model.to("cuda")
            
            self.model.eval()
            print("[OpenVLAAdapter] Model loaded successfully.")
            
        except Exception as e:
            print(f"[OpenVLAAdapter] Error loading model: {e}")
            print("Tip: Ensure you have installed: pip install transformers accelerate bitsandbytes")
            raise

    def predict(self, image, instruction: str) -> Dict:
        print(f"[OpenVLAAdapter] Predicting: '{instruction}'")
        
        # 预处理
        inputs = self.processor(instruction, image, return_tensors="pt")
        if self.backend_type == "nvidia":
            inputs = inputs.to("cuda", torch.bfloat16)
            
        # 推理
        with torch.no_grad():
             generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        # 解码
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # OpenVLA 输出通常是动作 tokens 或文本描述，需要进一步解析
        # 这里简单返回文本
        return {
            "raw_output": generated_text,
            "action": [0.0] * 7, # Mock parsed action
            "backend": self.backend_type
        }

