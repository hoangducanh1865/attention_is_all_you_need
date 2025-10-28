import torch
class Config:
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
