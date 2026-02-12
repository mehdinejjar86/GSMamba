# debug_grad_light.py
import torch
from config import get_full_config
from models.gs_mamba import build_model
from losses.combined import build_loss

device = torch.device('cuda')
config = get_full_config('gsmamba')
config.model.image_size = (256, 256)

# Build model and loss
model = build_model('gsmamba').to(device)
model.train()
criterion = build_loss(config.loss, config.model).to(device)

# Simulate REAL data (no requires_grad, like dataloader)
B, N, C, H, W = 2, 2, 3, 256, 256
frames = torch.rand(B, N, C, H, W, device=device)  # No requires_grad
target = torch.rand(B, C, H, W, device=device)      # No requires_grad
anchor_times = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device=device)
target_time = torch.tensor([0.5, 0.5], device=device)

print(f"frames requires_grad: {frames.requires_grad}")
print(f"target requires_grad: {target.requires_grad}")

# Forward
output = model(frames=frames, t=target_time, timestamps=anchor_times, return_intermediates=True)
print(f"pred requires_grad: {output['pred'].requires_grad}")

# Loss
losses = criterion(
    pred=output['pred'],
    target=target,
    render=output['render'],
    depth=output['depth'],
    input_frames=frames,
    gaussians_list=output.get('all_gaussians', []),
    gaussians_interp=output.get('gaussians', {}),
    t=0.5,
)

print(f"total requires_grad: {losses['total'].requires_grad}")
print(f"total grad_fn: {losses['total'].grad_fn}")

losses['total'].backward()
print("SUCCESS!")
