import os
import sys
import json
import argparse
from huggingface_hub import snapshot_download

# 设定项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "configs", "model_zoo.json")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def download_model(model_name, config):
    model_info = config['model_zoo'].get(model_name)
    if not model_info:
        print(f"Error: Model '{model_name}' not found in configuration.")
        print(f"Available models: {list(config['model_zoo'].keys())}")
        return

    if model_info.get('source') == 'manual':
        print(f"Skipping download for '{model_name}': Source is 'manual'.")
        print(f"Please manually place model files at: {model_info['local_path']}")
        return

    print(f"Preparing to download: {model_name}")
    print(f"  Repo: {model_info['repo_id']}")
    print(f"  Dest: {model_info['local_path']}")

    # 构造绝对路径
    local_dir = os.path.join(PROJECT_ROOT, model_info['local_path'])
    
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Warning: Directory {local_dir} is not empty. Skipping download (or force with --force).")
        # 实际逻辑中可以加 force 参数处理
        # return

    try:
        snapshot_download(
            repo_id=model_info['repo_id'],
            local_dir=local_dir,
            local_dir_use_symlinks=False, # 避免在 Windows 上出现 symlink 问题
            resume_download=True
        )
        print(f"Successfully downloaded {model_name} to {local_dir}")
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="BotMind Model Downloader")
    parser.add_argument("--model", type=str, help="Specific model name to download (e.g., x-vla-widowx)")
    parser.add_argument("--all", action="store_true", help="Download all models in config")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    config = load_config()

    if args.list:
        print("Available models in Model Zoo:")
        for name, info in config['model_zoo'].items():
            print(f" - {name}: {info['description']}")
        return

    if args.all:
        for name in config['model_zoo'].keys():
            download_model(name, config)
    elif args.model:
        download_model(args.model, config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

