from typing import Union
from datasets import load_dataset
from torch.utils.data import DataLoader
from tiny_dataset import TinyDataset


def get_data(
    batch_size: Union[int, None] = None,
    subset_size: Union[int, None] = None,
    seed: int = 42,
):
    train_dataset = load_dataset("Maysee/tiny-imagenet", split="train")
    val_dataset = load_dataset("Maysee/tiny-imagenet", split="valid")
    if subset_size is not None:
        train_dataset = train_dataset.shuffle(seed=seed).select(range(subset_size))
    train_data = TinyDataset(train_dataset)
    val_data = TinyDataset(val_dataset)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    train_dataloader, val_dataloader = get_data(batch_size=64, subset_size=1000, seed=42)
    for img, label in train_dataloader:
        print(img.shape, label.shape)
        break


