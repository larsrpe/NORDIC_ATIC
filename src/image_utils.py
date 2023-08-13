from typing import Tuple
from PIL import Image, ImageOps
import torch
from torchvision import transforms


def image_to_pdf_args(
    image: str, L: float, resolution: Tuple[int, int] = (128, 128)
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(image, str):
        image = Image.open(f"images/{image}.png")
    image = expand2square(image, "white")
    image = image.resize(resolution)
    image = image.resize(resolution)
    image = ImageOps.grayscale(image)
    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image)
    image = torch.flipud(image)
    image = 1 - image
    x_dim = image.shape[1]
    y_dim = image.shape[2]
    weights = image / torch.sum(image)
    weights = weights.reshape((x_dim, y_dim))

    x = torch.linspace(0, L, x_dim)
    y = torch.linspace(0, L, y_dim)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
    grid = torch.cat(
        (
            x_grid.reshape(x_dim, y_dim, 1),
            torch.flipud(y_grid).reshape(x_dim, y_dim, 1),
        ),
        dim=2,
    )
    return grid, weights


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
