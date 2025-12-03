import os
import numpy as np
import pickle
from .preprocessor import VideoPreprocessor

class DatasetBuilder:
    def __init__(self, output_dir="data/processed"):
        self.output_dir = output_dir
        self.preprocessor = VideoPreprocessor()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def build_from_videos(self, video_folder, dataset_name="custom_vla_data"):
        """
        扫描目录下的所有视频，构建用于 VLA 训练的 .npy/.pkl 数据集
        """
        print(f"[DatasetBuilder] Scanning {video_folder}...")
        all_episodes = []
        
        for root, dirs, files in os.walk(video_folder):
            for file in files:
                if file.endswith((".mp4", ".avi")):
                    video_path = os.path.join(root, file)
                    frames = self.preprocessor.extract_frames_from_video(video_path)
                    
                    # 模拟：通常还需要配对的 Action 数据
                    # 这里假设是自监督预训练（仅视频）或 Action 是分离加载的
                    episode = {
                        "video_path": video_path,
                        "frames": np.array(frames), # (T, H, W, C)
                        "actions": np.zeros((len(frames), 7)) # Dummy actions
                    }
                    all_episodes.append(episode)

        save_path = os.path.join(self.output_dir, f"{dataset_name}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(all_episodes, f)
        
        print(f"[DatasetBuilder] Saved {len(all_episodes)} episodes to {save_path}")
        return save_path

