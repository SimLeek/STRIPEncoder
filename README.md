# Sparse Trained Recurrent Image Pyramid Encoding (STRIPEncode)

STRIPEncode is a PyTorch-based library designed for real-time video compression using sparse, recurrent neural networks and image pyramids.

## Installation

Install STRIPEncode via pip:

```bash
pip install git+https://github.com/simleek/stripencode.git
```

## Quick Start

### Training a Video Compressor

Here's how you can train a video compressor with your webcam:

```python
import time
from torch import optim
from stripencode.webcam_capture import WebcamCapture
import cv2
import numpy as np
import torch
from stripencode import StripEncoder2D

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
```

## Todo

* Fix the border
  * I used 3x3 convolutions with padding because it was easy, but removing padding will fix the border. However, at small sizes when width or height is smaller than 3, the 3x3 needs to be reduced to width x height 

## Documentation

There is no documentation yet

## Contributing

Sure

## License

MIT License