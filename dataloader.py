from typing import Optional, Tuple, Callable
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from dataset import ImageDataset
from torchvision import transforms as T


def default_transforms(image_size: Tuple[int, int]) -> T.Compose:
    """Return default transform: resize and convert to tensor."""
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """Utility to create a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


class ImageDataModule(pl.LightningDataModule):
    """Lightning DataModule for image datasets.

    Args:
        train_dir: Path to training images.
        val_dir: Path to validation images.
        batch_size: Batch size for dataloaders.
        image_size: Tuple (H, W) for resizing images.
        num_workers: Number of workers for DataLoader.
        pin_memory: Pin memory in DataLoader.
        transform: Optional torchvision transform. Defaults to resize + ToTensor.
        drop_last: Whether to drop last incomplete batch in training.
        collate_fn: Optional custom collate function.
    """
    def __init__(
        self,
        train_dir: Optional[str],
        val_dir: Optional[str],
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        num_workers: int = 4,
        pin_memory: bool = True,
        transform: Optional[Callable] = None,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform or default_transforms(image_size)
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets. Called on every process by Lightning."""
        if self.train_dir and self.train_dataset is None:
            self.train_dataset = ImageDataset(self.train_dir, image_size=self.image_size, transform=self.transform)

        if self.val_dir and self.val_dataset is None:
            self.val_dataset = ImageDataset(self.val_dir, image_size=self.image_size, transform=self.transform)

    def train_dataloader(self) -> Optional[DataLoader]:
        if not self.train_dataset:
            return None
        return create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_dataset:
            return None
        return create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # never drop last batch in validation
            collate_fn=self.collate_fn,
        )
