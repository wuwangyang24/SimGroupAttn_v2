import os
from typing import Optional, Callable, Sequence, Iterator
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):
    """Dataset for loading images lazily from a directory with millions of files."""
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    def __init__(
        self,
        image_dir: str,
        image_size: Sequence[int] = (224, 224),
        transform: Optional[Callable] = None,
    ) -> None:
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Directory '{image_dir}' does not exist.")

        self.image_dir = image_dir
        self.transform = transform or T.Compose([
            T.Resize(tuple(map(int, image_size))),
            T.ToTensor()
        ])
        self._image_paths_cache: Optional[list[str]] = None  # lazy cache

    def _scan_image_paths(self) -> Iterator[str]:
        """Lazy generator over image paths in the directory."""
        with os.scandir(self.image_dir) as it:
            for entry in it:
                if entry.is_file() and not entry.name.startswith(".") \
                        and entry.name.lower().endswith(self.IMAGE_EXTENSIONS):
                    yield entry.path

    @property
    def image_paths(self) -> list[str]:
        """Cache the paths on first access."""
        if self._image_paths_cache is None:
            self._image_paths_cache = list(self._scan_image_paths())
            if not self._image_paths_cache:
                raise RuntimeError(f"No images found in '{self.image_dir}'.")
        return self._image_paths_cache

    def __len__(self) -> int:
        # For huge directories, len may be expensive on first call
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tensor:
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                return self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image '{img_path}': {exc}") from exc
