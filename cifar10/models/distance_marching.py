import torch.nn as nn
from models.unet import UNetModel


# ------------------------------------------------------------
# Distance head: a tiny CNN that predicts u_theta(x) (<30K params)
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
# Distance Marching model: distance head + UNet direction field
# ------------------------------------------------------------
class DistanceMarching(UNetModel):
    """UNet that predicts the direction v_theta(x), conditioned on the distance
    u_theta(x) predicted by a separate tiny CNN head (no time input).

    ``forward(x, timesteps, extra, return_dist=True)`` returns ``(grad, dist)``:
    ``grad`` points away from the data manifold (the samplers step along
    ``-grad``) and ``dist`` has shape [N, 1]. ``timesteps`` is ignored; it is
    only kept so that the training loop can use the usual model signature.
    """

    def __init__(self, *args, **kwargs):
        self.unified_scale = kwargs.pop("unified_scale", 1.0)

        in_channels = kwargs.get("in_channels", 3)
        image_size = kwargs.pop("image_size", (32, 32))
        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size
        self.image_dimension = in_channels * H * W

        # The UNet keeps its time-embedding MLP; it embeds the predicted distance.
        kwargs["ignore_time"] = False
        super().__init__(*args, **kwargs)

        # Created after the UNet so that the parameter order (and hence the EMA
        # shadow parameters of the released checkpoint) is unchanged.
        self.t_predictor = NoiseLevelPredictor(in_channels)

        t_predictor_params = sum(p.numel() for p in self.t_predictor.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print("=" * 60)
        print("DistanceMarching Model Initialized")
        print(f"  - distance head params: {t_predictor_params:,}")
        print(f"  - unified_scale: {self.unified_scale}")
        print(f"  - entries:      {self.image_dimension} (C={in_channels}, H={H}, W={W})")
        print(f"  - total params: {total_params:,}")
        print("=" * 60)

    def forward(self, x, timesteps, extra, return_dist=False):
        # (1) predict the distance to the data manifold
        t_pred = self.t_predictor(x).to(x.device)   # [N]

        # (2) the UNet receives x and the predicted distance and outputs the
        #     direction toward the data; negate it so that grad points away
        grad = -super().forward(x, t_pred, extra) / self.unified_scale

        if return_dist:
            return grad, t_pred.unsqueeze(-1)       # [N, 1]
        return grad
