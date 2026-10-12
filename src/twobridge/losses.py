"""Flow matching + optional I1a Sobolev loss; padding never contributes."""

import torch


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1)


def flow_loss(target, predicted, x_t, time, actions, mask, action_dim, deriv_dims, deriv_weight):
    mask = mask[..., None]
    flow = masked_mean((target[..., :action_dim] - predicted[..., :action_dim]).square(), mask)
    deriv = flow * 0
    if deriv_weight:
        clean = x_t - time[:, None, None] * predicted
        error = clean[..., deriv_dims] - actions[..., deriv_dims]
        for order, weight in ((1, 1.0), (2, 0.5)):
            valid = mask[:, : mask.shape[1] - order].clone()
            for shift in range(1, order + 1):
                valid &= mask[:, shift : mask.shape[1] - order + shift]
            values = torch.diff(error, n=order, dim=1).square()
            # Average per example first, matching the original time weighting.
            denom = valid.expand_as(values).sum(dim=(1, 2)).clamp_min(1)
            per_sample = (values * valid).sum(dim=(1, 2)) / denom
            deriv = deriv + weight * ((1 - time).clamp_min(0) * per_sample).mean()
    return {"loss": flow + deriv_weight * deriv, "flow_loss": flow, "deriv_loss": deriv}
