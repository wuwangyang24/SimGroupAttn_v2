import os
from typing import Optional, Callable, Sequence
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """Dataset for loading images from a directory."""
    def __init__(
        self,
        image_dir: str,
        image_size: Sequence[int] = (224, 224),
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = image_dir
        self.image_paths = []
        suffixes = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        try:
            with os.scandir(image_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if name.startswith("."):
                        continue
                    if name.lower().endswith(suffixes):
                        self.image_paths.append(os.path.join(image_dir, name))
        except FileNotFoundError:
            raise
        image_size = tuple(int(x) for x in image_size)
        self.transform = (
            transform
            if transform is not None
            else transforms.Compose([transforms.Resize(image_size), 
                                     transforms.ToTensor()])
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tensor:
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image '{img_path}': {exc}") from exc
        return tensor