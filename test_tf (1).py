try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("PyTorch available")
except ImportError:
    print("PyTorch not available")