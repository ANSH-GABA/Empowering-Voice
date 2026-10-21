import os
import kagglehub
from huggingface_hub import hf_hub_download

def download_all_assets():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Model/ml/data
    RAW_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "raw")) # Model/ml/data/raw

    print(f" Forcing local download to target project directory: {RAW_DATA_DIR}")

    print("Ingesting Kaggle ASL Alphabet Dataset...")
    dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    print(f"[+] Dataset available at: {dataset_path}")

    print("\nFetching pre-trained checkpoint binaries from HuggingFace...")
    model_path = hf_hub_download(
        repo_id="huzaifanasirrr/realtime-sign-language-translator",
        filename="best_model.pth"
    )
    
    mediapipe_path = hf_hub_download(
        repo_id="huzaifanasirrr/realtime-sign-language-translator",
        filename="hand_landmarker.task"
    )
    print(f"[+] Model checkpoint path: {model_path}")
    print(f"[+] MediaPipe task path: {mediapipe_path}")
    
    return dataset_path, model_path, mediapipe_path

if __name__ == "__main__":
    download_all_assets()