import torch, os, sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
sys.path.insert(0, '/home/exacloud/gscratch/ChangLab/govindsa/GSMamba')

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group('nccl', timeout=timedelta(hours=1))
torch.cuda.set_device(local_rank)

from models.gs_mamba import build_model
model = build_model('gsmamba_small').cuda()
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
model.train()

frames = torch.randn(2, 2, 3, 256, 256).cuda()
t = torch.tensor([0.5, 0.5]).cuda()
timestamps = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).cuda()
output = model(frames=frames, t=t, timestamps=timestamps, return_intermediates=True)

if rank == 0:
    print(f"pred requires_grad: {output['pred'].requires_grad}")
    print(f"pred grad_fn: {output['pred'].grad_fn}")
    try:
        output['pred'].sum().backward()
        print("Backward: SUCCESS")
    except Exception as e:
        print(f"Backward: FAILED — {e}")

dist.destroy_process_group()
