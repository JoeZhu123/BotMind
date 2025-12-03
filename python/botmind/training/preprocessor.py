import numpy as np
import cv2
import os

class VideoPreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def process_frame(self, frame_data, width, height):
        """
        将原始字节流转换为模型可用的 Tensor 格式 (H, W, C)
        """
        # 1. Convert raw bytes to numpy array
        img = np.array(frame_data, dtype=np.uint8).reshape((height, width, 3))
        
        # 2. Resize
        img_resized = cv2.resize(img, self.target_size)
        
        # 3. Normalize (0-1 or standard mean/std)
        img_norm = img_resized.astype(np.float32) / 255.0
        
        return img_norm

    def extract_frames_from_video(self, video_path, interval=10):
        """
        从视频文件提取帧，用于预训练数据集构建
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return frames

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if count % interval == 0:
                    # Resize and store
                    processed = cv2.resize(frame, self.target_size)
                    frames.append(processed)
                count += 1
            else:
                break
        
        cap.release()
        print(f"[Preprocessing] Extracted {len(frames)} frames from {video_path}")
        return frames

