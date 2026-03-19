import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from src.preprocessing.milho_preprocessing import get_transforms_milho

def get_dataloaders_milho(
    data_dir,
    batch_size=32,
    num_workers=0,
    pin_memory = torch.cuda.is_available()
):

    transform_train, transform_eval = get_transforms_milho()

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=transform_train
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=transform_eval
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=transform_eval
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )