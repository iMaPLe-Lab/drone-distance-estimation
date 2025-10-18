# python
import torch
print("is_available:", torch.cuda.is_available())
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)          # None means CPU-only wheel
print("cudnn:", torch.backends.cudnn.version())   # None if not available
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
