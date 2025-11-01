from typing import Optional, Tuple, Callable
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from dataset import ImageDataset
from torchvision import transforms as T

def default_transforms(image_size: Tuple[int, int]) -> T.Compose:
    """Return a simple default transform: resize and convert to tensor."""
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
    drop_last: bool,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """Utility function to create a DataLoader with given parameters."""
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
    """Lightning DataModule for image directories using ImageDataset.
    Example:
        dm = ImageDataModule(train_dir="/path/to/train", val_dir="/path/to/val", batch_size=64)
        trainer.fit(model, datamodule=dm)
    """
    def __init__(
        self,
        train_dir: Optional[str],
        val_dir: Optional[str],
        test_dir: Optional[str] = None,
        batch_size: int = 32,
        test_batch_size: int = 32,
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
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called by Lightning on every process."""
        trs = self.transform or default_transforms(self.image_size)

        if self.train_dir is not None and self.train_dataset is None:
            self.train_dataset = ImageDataset(self.train_dir, image_size=self.image_size, transform=trs)

        if self.val_dir is not None and self.val_dataset is None:
            self.val_dataset = ImageDataset(self.val_dir, image_size=self.image_size, transform=trs)

        if self.test_dir is not None and self.test_dataset is None:
            self.test_dataset = ImageDataset(self.test_dir, image_size=self.image_size, transform=trs)

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.train_dataset is None:
            return None
        return create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return create_dataloader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.collate_fn
        )

