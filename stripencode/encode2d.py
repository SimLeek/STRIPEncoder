import torch.nn.functional as F
import math
import torch
from torch import nn

def create_image_pyramid(x, num_levels=None, mode='bilinear'):
    """
    Given an input tensor x of shape [B, C, H, W], create a pyramid.
    At each level, the total pixel count is halved:
    Returns: list of tensors.
    """
    pyramid = [x]
    B, C, H, W = x.shape
    # scale factor per level: s = (1/√2) so that area is halved.
    scale_factor = 1 / math.sqrt(2)
    new_H = H
    new_W = W
    while pyramid[-1].shape[2] > 1 or pyramid[-1].shape[3] > 1:
        new_H = max(1, min(new_H - 1, int(new_H * scale_factor)))
        new_W = max(1, min(new_W - 1, int(new_W * scale_factor)))
        level = F.interpolate(x, size=(new_H, new_W), mode=mode, align_corners=False)
        pyramid.append(level)
    if num_levels is None:
        num_levels = len(pyramid)
    pyramid = pyramid[-num_levels:]

    return pyramid, num_levels


class RecurrentCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, image_size):
        super(RecurrentCNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.image_size = image_size
        # conv bias is true by default
        # we need that to backprop to the zero initialized hidden state from zero initialized weights
        self.conv = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.leaky_relu = nn.LeakyReLU()  # with normal ReLU, negative values get zeroed out and stop learning
        self.reset_hidden_state()

    def reset_hidden_state(self):
        # very useful. For eyes, use this whenever blinking or rapid movement
        self.h = torch.zeros(1, self.hidden_channels, *self.image_size)

    def forward(self, x):
        if self.h.device != x.device:
            self.h = self.h.to(x.device)
        x_h = torch.cat([x, self.h], dim=1)
        h_new = self.conv(x_h)
        self.conv.weight.data.clamp_(-2, 2)
        self.conv.bias.data.clamp_(-2, 2)
        h_new = self.leaky_relu(h_new)
        h_new = torch.clamp(h_new, -1, 1)
        self.h = h_new.detach()
        #return x + h_new
        out=F.pad(x, (0, 0, 0, 0, 0, h_new.shape[1] - x.shape[1], 0, 0)) + h_new
        out = torch.clamp(out, -1, 1)
        return out


class InverseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InverseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out = self.conv(x)
        # Clip weights to [-1, 1]
        self.conv.weight.data.clamp_(-2, 2)
        self.conv.bias.data.clamp_(-2, 2)
        # Skip connection
        out = x[:, :out.shape[1]] + out
        out = torch.clamp(out, -1, 1)
        return out


def normalize_image(image):
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    return image


def calc_image_pyramid_from_resolution(image_resolution, num_levels=None):
    """
    Given an image resolution (H, W), compute the pyramid levels.
    At each level the total pixel count is halved by scaling H and W by 1/sqrt(2).
    The pyramid continues until reaching 1x1.
    If num_levels is provided, only return the last num_levels levels (always including the 1x1 level).

    Returns:
        resolutions: List of (H, W) tuples for each level.
        actual_levels: Total number of levels in the pyramid.
    """
    H, W = image_resolution
    resolutions = [(H, W)]
    scale_factor = 1 / math.sqrt(2)
    while resolutions[-1][0] > 1 or resolutions[-1][1] > 1:
        prev_H, prev_W = resolutions[-1]
        new_H = max(1, int(prev_H * scale_factor))
        new_W = max(1, int(prev_W * scale_factor))
        # Prevent repeated same size if already at 1 in one dimension
        if (new_H, new_W) == resolutions[-1]:
            break
        resolutions.append((new_H, new_W))
    if num_levels is not None:
        # Always include the last (smallest) num_levels; discard earlier levels
        resolutions = resolutions[-num_levels:]
    return resolutions, len(resolutions)


def topk_percentage_loss(activations, k_ratio=0.1):
    """
    Computes an L1 sparsity loss that encourages the activations to match a target that retains
    only the top-k percentage of activations (per sample) and zeros out the rest.

    Args:
        activations (torch.Tensor): Input tensor of shape [B, C, H, W].
        k_ratio (float): Fraction of elements to keep (e.g., 0.1 for top 10%).

    Returns:
        loss (torch.Tensor): The averaged L1 loss over the batch.
    """
    B = activations.size(0)
    flat = activations.view(B, -1)  # shape: [B, total_elements]
    total_elements = flat.size(1)
    k = max(1, int(total_elements * k_ratio))

    loss = 0.0
    for i in range(B):
        sample = flat[i]  # shape: [total_elements]
        # Get top-k indices and values for this sample.
        topk_vals, topk_indices = torch.topk(sample, k=k, sorted=False)
        # Create a goal tensor: zero everywhere except at the top-k positions.
        goal = torch.zeros_like(sample)
        goal[topk_indices] = sample[topk_indices]
        # Compute the L1 difference between the original and the goal.
        loss += F.l1_loss(sample, goal, reduction='mean')
    return loss / B


def channel_ordering_loss(layer):
    """
    Enforces that for the averaged activations (over width and height) of a pyramid layer,
    channel i's activation is greater than or equal to channel i+1.

    Args:
        layer (torch.Tensor): Tensor of shape [B, C, H, W] from a pyramid level.
        weight (float): Scaling factor for this loss.

    Returns:
        loss (torch.Tensor): A scalar loss value.
    """
    # Average over spatial dimensions -> [B, C]
    A = torch.mean(layer, dim=(2, 3))
    # Compute differences: diff = A[:, :-1] - A[:, 1:]
    diff = A[:, :-1] - A[:, 1:]
    # For channels where A[i] is less than A[i+1], diff will be negative.
    # We incur loss equal to how much A[i+1] exceeds A[i]:
    ordering_violation = F.relu(-diff)  # zeros out cases where diff >= 0
    loss = ordering_violation.mean()
    return loss


class StripEncoder2D(nn.Module):
    def __init__(self, image_size, in_channels=3, downlayer_com_channels=3, recurrent_hidden_channels=2, num_pyramid_levels=None, quality=0.80, top_p=0.02,
                 order_weight=0.001):
        super(StripEncoder2D, self).__init__()
        adaptive_thresh_channels = 1
        self.quality = quality
        self.image_size = list(reversed(image_size))  # sane width x height to HW for NHWC
        self.resolutions, levels = calc_image_pyramid_from_resolution(self.image_size)
        self.num_pyramid_levels = num_pyramid_levels if num_pyramid_levels is not None else levels
        self.recurrent_cnns = nn.ModuleList([
            RecurrentCNN(in_channels, in_channels+adaptive_thresh_channels+downlayer_com_channels+recurrent_hidden_channels, self.resolutions[l])
            for l in range(self.num_pyramid_levels)
        ])
        self.inverse_convs = nn.ModuleList([
            InverseConv(in_channels+recurrent_hidden_channels, in_channels)
            for _ in range(self.num_pyramid_levels)
        ])
        self.top_p = top_p
        self.order_weight = order_weight
        self.in_channels = in_channels
        self.adaptive_thresh_channels = adaptive_thresh_channels
        self.downlayer_com_channels=downlayer_com_channels
        self.recurrent_hidden_channels = recurrent_hidden_channels

    def forward(self, x):
        x = normalize_image(x)
        pyr, levels = create_image_pyramid(x, self.num_pyramid_levels)
        compressed_pyramid = []
        h_com_out = None
        for l in range(self.num_pyramid_levels):
            if h_com_out is not None:
                h_com_in = F.interpolate(h_com_out, size=tuple(self.resolutions[l]), mode='bilinear')
                if self.recurrent_cnns[l].h.device != x.device:
                    self.recurrent_cnns[l].h = self.recurrent_cnns[l].h.to(x.device)
                self.recurrent_cnns[l].h[:, self.in_channels+self.adaptive_thresh_channels:self.in_channels+self.adaptive_thresh_channels+self.downlayer_com_channels, ...] = h_com_in
            h = self.recurrent_cnns[l](pyr[l])
            h_com_out = h[:, self.in_channels+self.adaptive_thresh_channels:self.in_channels+self.adaptive_thresh_channels+self.downlayer_com_channels, ...]
            compressed_pyramid.append(h)
        # Apply sparsity to compressed_pyramid
        sparse_pyramid = []
        for compressed_level, h in zip(compressed_pyramid, self.recurrent_cnns):
            threshold = h.h[:, self.in_channels:self.in_channels+self.adaptive_thresh_channels, :, :]  # Use first non-skip channel of hidden state as threshold
            sparse_level = torch.where(compressed_level >= threshold, compressed_level,
                                       torch.zeros_like(compressed_level))
            #sparse_pyramid.append(sparse_level)
            sparse_pyramid.append(torch.cat((sparse_level[:, :self.in_channels, ...], sparse_level[:, self.in_channels+self.adaptive_thresh_channels+self.downlayer_com_channels:, ...]), 1))

        reconstructed = torch.zeros_like(x)
        for l in range(self.num_pyramid_levels):
            recon = self.inverse_convs[l](sparse_pyramid[l])
            recon = F.interpolate(recon, size=tuple(self.image_size), mode='bilinear')
            reconstructed += recon
        reconstructed /= self.num_pyramid_levels


        kl_losses = sum(topk_percentage_loss(level, self.top_p) for level in sparse_pyramid)
        order_loss = sum(channel_ordering_loss(level) for level in sparse_pyramid)
        reproduction_loss = F.mse_loss(reconstructed, x)

        reconstructed = torch.clamp(reconstructed, 0, 1)

        # Combine losses with the given quality ratio
        total_loss = self.quality * reproduction_loss + (1 - self.quality) * kl_losses + order_loss * self.order_weight

        # Return sparse representation
        reversed_hidden = list(reversed(sparse_pyramid))
        unrolled = [level.view(level.shape[0], -1) for level in reversed_hidden]
        concatenated = torch.cat(unrolled, dim=1)
        #sp = csr_matrix(concatenated.detach().cpu().numpy())
        sparse_h = concatenated.to_sparse_csr()
        # force int32 because int64 is too big even for us
        crow = sparse_h.crow_indices().to(torch.int32)
        col = sparse_h.col_indices().to(torch.int32)
        values = sparse_h.values()
        # Reconstruct the sparse CSR tensor with int32 indices:
        sparse_h = torch.sparse_csr_tensor(crow, col, values, size=sparse_h.size())

        #sparse_h = torch.sparse_csr_tensor(sp.indptr, sp.indices, sp.data)

        # Image pyramid info for potential reconstruction
        pyramid_info = [(level.shape[1], level.shape[2], level.shape[3]) for level in sparse_pyramid]

        return sparse_h, pyramid_info, total_loss, reconstructed


if __name__ == '__main__':
    import time
    from torch import optim
    from stripencode.webcam_capture import WebcamCapture
    import cv2
    import numpy as np

    resolution = (640, 360)

    # Initialize model, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StripEncoder2D(in_channels=3, downlayer_com_channels=3, recurrent_hidden_channels=2, image_size=resolution).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3)  # Smaller LR for longer training cycles

    # Training loop with while loop (user can exit with Ctrl + C)
    step = 0
    try:
        with WebcamCapture(*resolution, 0) as webcam:
            while True:
                start_time = time.time()

                frame_tensor = webcam.get_frame(resolution)

                image = frame_tensor.to(device)
                sparse_h, pyramid_info, loss, reproduction = model(image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute memory savings
                original_size = image.element_size() * image.nelement()  # Uncompressed tensor size
                sparse_size = sparse_h.col_indices().numel()*sparse_h.col_indices().element_size()+ sparse_h.values().numel()*sparse_h.values().element_size()  # Only non-zero elements, *2 because indices
                compression_ratio = sparse_size / original_size

                print(
                    f"Step {step} | Loss: {loss.item():.6f} | Compression Ratio: {compression_ratio:.4f}'{sparse_h._nnz()}/{image.numel()} | Time per step: {time.time() - start_time:.3f}s")

                o = (frame_tensor.detach().cpu().squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
                r = (reproduction.detach().cpu().squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
                cv2.imshow("Original | Reproduction", np.hstack([o, r]))
                cv2.waitKey(1)
                step += 1

    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
