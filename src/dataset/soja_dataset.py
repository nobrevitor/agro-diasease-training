"""
soja_dataset.py

Módulo responsável por criar transforms, datasets ImageFolder
e DataLoaders para o projeto de visão computacional com soja.
"""

from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATASET_SPLIT = Path("/Volumes/workspace/agro-diasease/soja/dataset_soja_split")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# TRANSFORMS

def criar_transform_treino(
    img_size: int = IMG_SIZE
):
    """
    Cria as transformações para o conjunto de treino.

    Inclui augmentação online.
    """

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def criar_transform_validacao_teste(
    img_size: int = IMG_SIZE
):
    """
    Cria as transformações para validação e teste.

    Não aplica augmentação.
    """

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def criar_transforms(
    img_size: int = IMG_SIZE
):
    """
    Cria os transforms de treino, validação e teste.
    """

    transform_treino = criar_transform_treino(img_size)
    transform_val = criar_transform_validacao_teste(img_size)
    transform_teste = criar_transform_validacao_teste(img_size)

    return transform_treino, transform_val, transform_teste


# DATASETS

def criar_dataset_imagefolder(
    caminho: Path,
    transform=None
):
    """
    Cria um dataset ImageFolder a partir de uma pasta.
    """

    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {caminho}")

    return datasets.ImageFolder(
        root=str(caminho),
        transform=transform
    )


def criar_datasets(
    pasta_split: Path = DATASET_SPLIT,
    img_size: int = IMG_SIZE
):
    """
    Cria datasets de treino, validação e teste.
    """

    pasta_split = Path(pasta_split)

    transform_treino, transform_val, transform_teste = criar_transforms(
        img_size=img_size
    )

    dataset_treino = criar_dataset_imagefolder(
        caminho=pasta_split / "train",
        transform=transform_treino
    )

    dataset_val = criar_dataset_imagefolder(
        caminho=pasta_split / "val",
        transform=transform_val
    )

    dataset_teste = criar_dataset_imagefolder(
        caminho=pasta_split / "test",
        transform=transform_teste
    )

    return dataset_treino, dataset_val, dataset_teste


# DATALOADERS

def criar_dataloader(
    dataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool | None = None
):
    """
    Cria um DataLoader para um dataset.
    """

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def criar_dataloaders(
    pasta_split: Path = DATASET_SPLIT,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
):
    """
    Cria os datasets e dataloaders de treino, validação e teste.
    """

    dataset_treino, dataset_val, dataset_teste = criar_datasets(
        pasta_split=pasta_split,
        img_size=img_size
    )

    loader_treino = criar_dataloader(
        dataset=dataset_treino,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    loader_val = criar_dataloader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    loader_teste = criar_dataloader(
        dataset=dataset_teste,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return loader_treino, loader_val, loader_teste


# INFORMAÇÕES DAS CLASSES

def obter_classes(dataset):
    """
    Retorna a lista de classes do dataset.
    """

    return dataset.classes


def obter_class_to_idx(dataset):
    """
    Retorna o mapeamento classe -> índice.
    """

    return dataset.class_to_idx


def obter_idx_to_class(dataset):
    """
    Retorna o mapeamento índice -> classe.
    """

    return {
        indice: classe
        for classe, indice in dataset.class_to_idx.items()
    }


def mostrar_info_dataset(dataset, nome: str = "Dataset") -> None:
    """
    Mostra informações básicas de um dataset ImageFolder.
    """

    print(f"\n{nome}")
    print(f"Total de imagens: {len(dataset)}")
    print(f"Total de classes: {len(dataset.classes)}")
    print("Classes:")

    for indice, classe in enumerate(dataset.classes):
        print(f"{indice}: {classe}")


def mostrar_info_datasets(
    dataset_treino,
    dataset_val,
    dataset_teste
) -> None:
    """
    Mostra informações dos datasets de treino, validação e teste.
    """

    mostrar_info_dataset(dataset_treino, "Dataset de treino")
    mostrar_info_dataset(dataset_val, "Dataset de validação")
    mostrar_info_dataset(dataset_teste, "Dataset de teste")

if __name__ == "__main__":
    print(
        "Este arquivo foi criado para ser importado em notebooks.\n\n"
        "Exemplo:\n"
        "from soja_dataset import criar_datasets, criar_dataloaders\n"
    )