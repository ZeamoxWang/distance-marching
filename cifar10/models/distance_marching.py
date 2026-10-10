import torch
import torch.nn as nn
import math
from models.unet import UNetModel


# ------------------------------------------------------------
# Tiny Noise Level Predictor (<30K params)
# ------------------------------------------------------------
class NoiseLevelPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h).squeeze(-1)   # [N]


# ------------------------------------------------------------
# Main DistanceMarching Class
# ------------------------------------------------------------
class DistanceMarching(UNetModel):
    def __init__(self, *args, **kwargs):
        # Read external argument
        self.separate_t = kwargs.pop("separate_t", False)
        self.unified_scale = kwargs.pop("unified_scale", 1.0)

        in_channels = kwargs.get("in_channels", 3)
        image_size = kwargs.pop("image_size", (32, 32))

        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size

        self.image_dimension = in_channels * H * W
        self.sqrt_image_dimension = math.sqrt(self.image_dimension)
        self.prev_state = None

        # --------------------------------------------
        # Conditional ignore_time logic
        # --------------------------------------------
        # If separate_t=False → old behavior → ignore_time=True
        # If separate_t=True  → we want t_pred to matter → ignore_time=False
        # --------------------------------------------
        if self.separate_t:
            kwargs["ignore_time"] = False
        else:
            kwargs["ignore_time"] = True

        # Initialize parent UNet
        super().__init__(*args, **kwargs)

        # Tiny t predictor when needed
        if self.separate_t:
            self.t_predictor = NoiseLevelPredictor(in_channels)
            t_predictor_params = sum(p.numel() for p in self.t_predictor.parameters())
            print("[DistanceMarching] separate_t=True → ignore_time=False → Using t predictor")
            print(f"[DistanceMarching] NoiseLevelPredictor params: {t_predictor_params:,}")
        else:
            print("[DistanceMarching] separate_t=False → ignore_time=True → Original DM logic")

        print("=" * 60)
        print("DistanceMarching Model Initialized")
        print(f"  - separate_t:   {self.separate_t}")
        print(f"  - ignore_time:  {kwargs['ignore_time']}")
        print(f"  - unified_scale: {self.unified_scale}")
        print(f"  - entries:      {self.image_dimension} (C={in_channels}, H={H}, W={W})")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  - total params: {total_params:,}")
        print("=" * 60)


    # ------------------------------------------------------------
    # Original distance logic (unchanged)
    # ------------------------------------------------------------
    def forward_dist(self, x, timesteps, extra):
        batch_size = x.shape[0]
        denoising = super().forward(x, timesteps, extra).reshape(batch_size, -1)
        x_reshaped = x.reshape(batch_size, -1)
        dist = (x_reshaped * denoising).sum(dim=1, keepdim=True) / self.unified_scale
        return dist


    # ------------------------------------------------------------
    # Unified forward()
    # ------------------------------------------------------------
    def forward(self, x, timesteps, extra, return_dist=False):

        # ---------------------------------------------
        # Case 1: separate_t=False → original DM logic
        # ---------------------------------------------
        if not self.separate_t:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                dist = self.forward_dist(x, timesteps, extra)
                grad = torch.autograd.grad(dist.sum(), x, create_graph=self.training)[0]

            if return_dist:
                return grad, dist
            else:
                return grad

        else:
            # ---------------------------------------------
            # Case 2: separate_t=True → NEW logic
            # ---------------------------------------------
            # (1) predict t
            t_pred = self.t_predictor(x).to(x.device)   # [N]

            # (2) UNet receives x and the predicted t
            #     UNet **directly outputs the grad**
            grad = -super().forward(x, t_pred, extra) / self.unified_scale
            # UNet points backward to x, so we need to negate the gradient

            # (3) Output package: if return_dist=True, return t_pred as "dist"
            if return_dist:
                return grad, t_pred.unsqueeze(-1)           # [N, 1]

            return grad