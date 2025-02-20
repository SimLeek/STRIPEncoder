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

    resolution = (1280, 720)

    # Initialize model, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyramidCompressionNet(in_channels=3, hidden_channels=16, image_size=resolution).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-6)  # Smaller LR for longer training cycles

    # Training loop with while loop (user can exit with Ctrl + C)
    step = 0
    try:
        with WebcamCapture(*resolution, 0) as webcam:
            while True:
                start_time = time.time()

                frame_tensor = webcam.get_frame(resolution)

                image = frame_tensor.to(device)
                sparse_h, pyramid_info, loss = model(image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute memory savings
                original_size = image.element_size() * image.nelement()  # Uncompressed tensor size
                sparse_size = sparse_h._nnz() * sparse_h.element_size()  # Only non-zero elements
                compression_ratio = sparse_size / original_size

                print(
                    f"Step {step} | Loss: {loss.item():.6f} | Compression Ratio: {compression_ratio:.4f} | Time per step: {time.time() - start_time:.3f}s")

                step += 1

    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
```

## Documentation

There is no documentation yet

## Contributing

Sure

## License

MIT License