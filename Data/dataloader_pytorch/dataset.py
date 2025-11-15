import json
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.io import read_file, decode_image, ImageReadMode
from typing import Tuple, List, Dict, Any, Optional


# --------- helpers ---------
def _list_images(root: Path, exts: Tuple[str, ...]) -> List[str]:
    # single pass, fast scandir-based crawl
    stack = [root]
    out = []
    exts = set(e.lower() for e in exts)
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    suf = os.path.splitext(e.name)[1].lower()
                    if suf in exts:
                        out.append(e.path)
    out.sort()
    return out

def _filelist_cache_path(root: Path) -> Path:
    return root / ".filelist.json"

def _build_filelist(root: Path, exts: Tuple[str, ...]) -> List[str]:
    # We store: {"root_mtime": float, "files": [...]}
    cache = _filelist_cache_path(root)
    try:
        st = root.stat().st_mtime
        if cache.exists():
            obj = json.loads(cache.read_text())
            if abs(obj.get("root_mtime", 0.0) - st) < 1e6 and obj.get("files"):
                return obj["files"]
        # rebuild
        files = _list_images(root, exts)
        cache.write_text(json.dumps({"root_mtime": st, "files": files}))
        return files
    except Exception:
        # fallback (don’t crash training on cache errors)
        return _list_images(root, exts)

class ImageFolderDataset(Dataset):
    """
    ImageFolder-style dataset with:
      - cached file list (no per-epoch rglob)
      - torchvision.io decode (no PIL)
      - resize on tensor
      - uint8 until collate for vectorized conversion
      - returns dict: {"pixel_values": uint8 [C,H,W], "class": int, "path": str}
    """
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        self.root = Path(root)
        self.extensions = extensions
        self.files: List[str] = _build_filelist(self.root, self.extensions)
        if not self.files:
            raise ValueError(
                f"No images found under {self.root} with extensions {self.extensions}"
            )

        self.image_size = image_size
        self._interpolation = InterpolationMode.BILINEAR

        # infer classes from first directory level under root
        class_names = set(
            Path(p).relative_to(self.root).parts[0]
            for p in self.files
        )
        self.classes: List[str] = sorted(class_names)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.files)

    def _load_and_resize(self, path: str) -> torch.Tensor:
        # read & decode to CHW uint8
        data = read_file(path)
        # decode_image handles jpg/png/webp/bmp, returns CHW uint8 by default
        img = decode_image(data, mode=ImageReadMode.RGB)  # [C,H,W], uint8
        # resize (kept uint8 for speed), antialias True for quality
        img = TF.resize(
            img,
            [self.image_size, self.image_size],
            interpolation=self._interpolation,
            antialias=True,
        )
        return img  # uint8 0..255

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        img = self._load_and_resize(path)

        rel = Path(path).relative_to(self.root)
        class_name = rel.parts[0]
        label = self.class_to_idx[class_name]

        return {
            "pixel_values": img,    # uint8 [C,H,W], 0..255
            "class": label,         # int
            "path": path,           # str (handy for debugging)
        }