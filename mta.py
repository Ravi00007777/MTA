import torch
import torch.nn.functional as F 
from contextlib import nullcontext

def gaussian_kernel(mu, bandwidth, datapoints):
    # bandwidth can be scalar or per-view; clamp for numerical stability.
    bandwidth = bandwidth.clamp_min(1e-6)
    dist = torch.norm(datapoints - mu, dim=-1, p=2)
    density = torch.exp(-dist**2 / (2 * bandwidth**2))
    return density


def solve_mta(model, inputs, args):
    amp_ctx = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext
    with torch.no_grad():
        with amp_ctx():
            image_features, text_features, logit_scale = model(inputs, features=True)
    logits = image_features @ text_features.t() * logit_scale

    lambda_y = args.lambda_y
    lambda_q = args.lambda_q
    max_iter = args.mta_max_iter
    temperature = args.mta_tau
    tol = args.mta_tol

    batch_size = image_features.shape[0]

    # Adaptive bandwidth from neighborhood distances.
    dist = torch.cdist(image_features, image_features)
    sorted_dist, _ = torch.sort(dist, dim=1)
    k = max(1, int(args.mta_bandwidth_frac * (batch_size - 1)))
    selected_distances = sorted_dist[:, 1:k+1]**2  # exclude the distance to the point itself 
    mean_distance = torch.mean(selected_distances, dim=1)
    bandwidth = torch.sqrt(0.5 * mean_distance).clamp_min(1e-6)

    # Text-based affinity: agreement in class-probability space (temperature-scaled).
    probs = (logits / temperature).softmax(1)
    affinity_matrix = probs @ probs.t()

    # Inlierness initialization on simplex.
    y = torch.ones(batch_size, device=image_features.device) / batch_size

    # Weighted embedding initialization (instead of single-view init).
    mode = torch.sum(y.unsqueeze(1) * image_features, dim=0)
    mode = mode / mode.norm(p=2, dim=-1, keepdim=False).clamp_min(1e-6)

    for _ in range(max_iter):
        old_mode = mode
        old_y = y

        # Eq. 6: inlierness update (softmax keeps y on probability simplex).
        density = gaussian_kernel(mode, bandwidth, image_features)
        weighted_affinity = affinity_matrix * y.unsqueeze(0)
        y = F.softmax((density + lambda_q * torch.sum(weighted_affinity, dim=1)) / lambda_y, dim=-1)

        # Eq. 8: kernel MeanShift mode update with soft inlierness weights.
        density = gaussian_kernel(mode, bandwidth, image_features)
        weighted_density = density * y
        denom = torch.sum(weighted_density).clamp_min(1e-6)
        mode = torch.sum(weighted_density.unsqueeze(1) * image_features, dim=0) / denom
        mode = mode / mode.norm(p=2, dim=-1, keepdim=False).clamp_min(1e-6)

        if torch.norm(old_mode - mode, p=2) < tol and torch.norm(old_y - y, p=2) < tol:
            break

    output = mode.unsqueeze(0) @ text_features.t() * logit_scale
    return output
