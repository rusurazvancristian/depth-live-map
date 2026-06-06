@echo off
echo Installing PyTorch with CUDA 12.1 (RTX 3060)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing remaining dependencies...
pip install transformers>=4.38.0 Pillow>=10.0.0 tqdm>=4.65.0 matplotlib>=3.7.0 huggingface_hub

echo Done. Run: python D:\mental_map_slam\main.py
pause
