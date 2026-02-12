# debug_grad.py
import torch
import torch.nn as nn
from config import get_full_config
from models.gs_mamba import build_model
from losses.combined import build_loss

# Setup
device = torch.device('cuda')
config = get_full_config('gsmamba')

# Build model and loss
model = build_model('gsmamba').to(device)
criterion = build_loss(config.loss, config.model).to(device)

# Create dummy data
B, N, C, H, W = 2, 2, 3, 256, 256
frames = torch.randn(B, N, C, H, W, device=device)
target = torch.randn(B, C, H, W, device=device)
anchor_times = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device=device)
target_time = torch.tensor([0.5, 0.5], device=device)

# Forward pass
model.train()
output = model(frames=frames, t=target_time, timestamps=anchor_times, return_intermediates=True)

print("=== Checking requires_grad ===")
print(f"pred requires_grad: {output['pred'].requires_grad}")
print(f"render requires_grad: {output['render'].requires_grad}")
print(f"depth requires_grad: {output['depth'].requires_grad}")

if 'all_gaussians' in output and len(output['all_gaussians']) > 0:
    g = output['all_gaussians'][0]
    for k, v in g.items():
        print(f"gaussians[0]['{k}'] requires_grad: {v.requires_grad}")

# Try loss computation
print("\n=== Computing loss ===")
losses = criterion(
    pred=output['pred'],
    target=target,
    render=output['render'],
    depth=output['depth'],
    input_frames=frames,
    gaussians_list=output.get('all_gaussians', []),
    gaussians_interp=output.get('gaussians', {}),
    t=target_time.mean().item(),
)

print(f"total loss requires_grad: {losses['total'].requires_grad}")
print(f"total loss grad_fn: {losses['total'].grad_fn}")

for k, v in losses.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: requires_grad={v.requires_grad}, grad_fn={v.grad_fn is not None}")

# Try backward
print("\n=== Trying backward ===")
losses['total'].backward()
print("Backward succeeded!")
