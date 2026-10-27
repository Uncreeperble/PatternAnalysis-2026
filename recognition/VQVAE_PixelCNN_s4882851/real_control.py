"""Real-to-real control for the nearest-reference SSIM protocol.

For each real TEST slice, find the nearest real VALIDATE slice by pixel L2
(patient-disjoint splits) and compute SSIM on that pair - exactly mirroring
the generated-image protocol in predict.py. If real scans also score far
below 0.6 here, the protocol (unpaired anatomy matching), not the generator,
explains the low generated scores.
"""
import json
import sys

import torch
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, "/home/Student/s4882851/PatternAnalysis-2026/recognition/VQVAE_PixelCNN_s4882851")
from dataset import DEFAULT_ROOT, HipMRISlices  # noqa: E402

test = HipMRISlices(DEFAULT_ROOT, "test")._cache          # [N,1,256,128]
ref = HipMRISlices(DEFAULT_ROOT, "validate")._cache
q = test.flatten(1)
r = ref.flatten(1)
# Nearest validate slice by L2 for every test slice (same rule as predict.py).
d = (q.square().sum(1, keepdim=True) + r.square().sum(1) - 2 * q @ r.t()).clamp_min(0)
nearest = d.argmin(1)
scores = [float(ssim(test[i, 0].numpy(), ref[int(j), 0].numpy(), data_range=1.0))
          for i, j in enumerate(nearest)]
t = torch.tensor(scores)
result = {
    "protocol": "real test slice vs nearest real validate slice by pixel L2, SSIM",
    "n": len(scores),
    "mean_ssim": float(t.mean()),
    "median_ssim": float(t.median()),
    "fraction_ge_0.6": float((t >= 0.6).float().mean()),
    "min": float(t.min()),
    "max": float(t.max()),
}
print(json.dumps(result, indent=2))
with open("/home/Student/s4882851/real_control.json", "w") as f:
    json.dump(result, f, indent=2)