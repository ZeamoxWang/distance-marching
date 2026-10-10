# Distance Marching training losses in the latent space (ImageNet).
#
# The model receives x_t = t * x_1 + (1 - t) * x_0 (x_1: VAE latent of an
# image, x_0: Gaussian noise, t ~ U(0, 1 - eps)) and returns the rescaled
# direction toward the data together with the predicted distance. The losses
# are the one-step loss with inverse-squared-distance reweighting and the
# directional eikonal loss of the paper.
import torch


def expand_t_like_x(t, x):
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


def sample_path(x1, train_eps: float = 1e-3):
    """Sample t ~ U(0, 1 - train_eps), x_0 ~ N(0, I) and return (t, x_0, x_t, u_t)."""
    x0 = torch.randn_like(x1)
    t0, t1 = 0.0, 1.0 - train_eps
    t = torch.rand((x1.shape[0],)) * (t1 - t0) + t0
    t = t.to(x1)
    tt = expand_t_like_x(t, x1)
    alpha_t, d_alpha_t = tt, 1
    sigma_t, d_sigma_t = 1 - tt, -1
    xt = alpha_t * x1 + sigma_t * x0
    ut = d_alpha_t * x1 + d_sigma_t * x0
    return t, x0, xt, ut


def training_losses(
    model,
    x1,
    y,
    one_step_loss_weight: float = 1.0,
    eikonal_loss_weight: float = 30.0,
    c: float = 100.0,
    train_eps: float = 1e-3,
):
    """Returns a dict with 'loss' (per-sample) and monitoring terms.

    Args:
        model: (DDP-wrapped) DistanceMarchingSiT. Called with get_energy=True it
            returns (direction, -distance).
        x1: latents, shape [B, 4, 32, 32].
        y: class labels, shape [B].
        one_step_loss_weight: lambda_OS.
        eikonal_loss_weight: lambda_DE.
        c: epsilon in the one-step loss and c_0 in the directional eikonal loss.
    """
    t, x0, xt, ut = sample_path(x1, train_eps=train_eps)

    model_output, dist = model(xt, t, y=y, train=True, get_energy=True)
    # the gradients in the DIST model should point towards the noise
    model_output = -model_output

    B = xt.shape[0]
    terms = {}

    ut_norm = torch.norm(ut.reshape(B, -1), dim=1, keepdim=False)
    smooth_norm = torch.sqrt(((1 - t) * ut_norm) ** 2 + c)
    eikonal_loss = (
        (model_output + ut * (1 - t).reshape(B, 1, 1, 1) / smooth_norm.reshape(B, 1, 1, 1))
        ** 2
    ).reshape(B, -1).sum(dim=1)
    one_denoising_step = -model_output * dist.reshape(B, 1, 1, 1)
    one_step_error = (xt + one_denoising_step - x1) / smooth_norm.reshape(B, 1, 1, 1)
    one_step_loss = (one_step_error**2).reshape(B, -1).sum(dim=1)

    # the real distance is (1 - t) * ||u_t||
    real_distance = (1 - t) * ut_norm
    ref_distance = smooth_norm
    one_denoising_step_norm = torch.norm(one_denoising_step.reshape(B, -1), dim=1, keepdim=False)
    real_distance_MAPE = one_denoising_step_norm / real_distance - 1
    ref_distance_MAPE = dist / ref_distance - 1

    terms["eikonal_loss"] = eikonal_loss.mean()
    terms["one_step_loss"] = one_step_loss.mean()
    terms["real_distance_MAPE"] = real_distance_MAPE.abs().mean()
    terms["ref_distance_MAPE"] = ref_distance_MAPE.abs().mean()
    terms["loss"] = eikonal_loss_weight * eikonal_loss + one_step_loss_weight * one_step_loss
    return terms
