import cv2
import torch

class WebcamCapture:
    def __init__(self, width=1280, height=720, device=0):
        self.device = device
        self.width = width
        self.height = height
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        return self

    def get_frame(self, resolution):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from webcam")
        # Resize and normalize the image
        frame = cv2.resize(frame, resolution) / 255.0
        # Convert the image from H x W x C to a tensor of shape [B, C, H, W]
        tensor_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return tensor_frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()