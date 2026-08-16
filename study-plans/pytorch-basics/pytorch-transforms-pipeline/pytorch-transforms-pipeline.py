import torch

class TransformPipeline:
    """
    Returns: float32 tensor of shape (C, H, W) from __call__
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, image):
        image_norm = image.float() / 255.0
        image_re = image_norm.permute(2, 0, 1)
        x = (image_re - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        return x
