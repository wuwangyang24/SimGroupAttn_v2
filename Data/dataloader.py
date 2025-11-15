import os
import torch
import hashlib
import lightning as pl
from typing import Tuple, List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from dataset import ImageFolderDataset


def _hash_split(path: str, val_ratio: float) -> bool:
    # deterministic split by hashing path (stable across runs/machines)
    h = int(hashlib.md5(path.encode("utf-8")).hexdigest(), 16)
    return (h % 10_000) < int(10_000 * val_ratio)


# --------- collate: uint8 -> float32 & stack labels ---------
def fast_supervised_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    batch: list of {"pixel_values": uint8 [C,H,W], "class": int, ...}
    returns:
      {
        "pixel_values": float32 [B,C,H,W] in [0,1],
        "class": int64 [B]
      }
    """
    imgs = [b["pixel_values"] for b in batch]
    labels = [b["class"] for b in batch]
    batch_uint8 = torch.stack(imgs, dim=0)  # BCHW, uint8
    pixel_values = batch_uint8.to(torch.float32).div_(255.0)
    class_tensor = torch.tensor(labels, dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "class": class_tensor,
    }


# --------- LightningDataModule ---------
class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        val_split: float = 0.1,
        image_size: int = 224,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        num_workers: Optional[int] = None,
        pin_memory_device: Optional[str] = "cuda",
        prefetch_factor: int = 4,
    ):
        """
        Lightning wrapper around ImageFolderDataset + fast_supervised_collate + hash-based split.
        Batches look like: {"pixel_values": float32[B,C,H,W], "class": int64[B]}.
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.image_size = image_size
        self.extensions = extensions
        if num_workers is None:
            num_workers = min(16, os.cpu_count() or 4)
        self.num_workers = num_workers
        self.pin_memory_device = pin_memory_device
        self.prefetch_factor = prefetch_factor
        self._full_ds: Optional[ImageFolderDataset] = None
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.save_hyperparameters(ignore=["pin_memory_device"])

    def setup(self, stage: Optional[str] = None):
        if self._full_ds is not None:
            return

        self._full_ds = ImageFolderDataset(
            root=self.root,
            image_size=self.image_size,
            extensions=self.extensions,
        )
        files = self._full_ds.files

        # Deterministic train/val split via hashing
        val_paths = set(p for p in files if _hash_split(p, self.val_split))
        train_paths = [p for p in files if p not in val_paths]
        val_paths = list(val_paths)

        # Lightweight view datasets reusing same logic
        class _View(ImageFolderDataset):
            def __init__(self, base: ImageFolderDataset, subset: List[str]):
                # reuse everything except 'files'
                self.root = base.root
                self.extensions = base.extensions
                self.files = subset
                self.image_size = base.image_size
                self._interpolation = base._interpolation
                self.classes = base.classes
                self.class_to_idx = base.class_to_idx

        self.train_ds = _View(self._full_ds, train_paths)
        self.val_ds = _View(self._full_ds, val_paths)

    def _common_dataloader_kwargs(self) -> dict:
        kwargs = dict(
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.num_workers and self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = self.prefetch_factor
        if self.pin_memory_device is not None:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return kwargs

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Call setup() before train_dataloader().")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=fast_supervised_collate,
            **self._common_dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Call setup() before val_dataloader().")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=fast_supervised_collate,
            **self._common_dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        # reuse val split for test, or adjust if you have a separate test folder
        return self.val_dataloader()
