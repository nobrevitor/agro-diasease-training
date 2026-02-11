# ==========================================================
# PREPROCESSAMENTO - DATASET MILHO
# Split + Data Augmentation + DataLoaders
# ==========================================================

import os
import shutil
import zipfile
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================================
# 1️⃣ EXTRAÇÃO DO DATASET (se necessário)
# ==========================================================

def extrair_dataset_milho(zip_path, extract_path):
    """
    Extrai o dataset apenas se a pasta ainda não existir.
    """
    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print("Dataset já extraído. Pulando extração.")
        return

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("Dataset extraído com sucesso!")


# ==========================================================
# 2️⃣ VERIFICAÇÃO SE O SPLIT JÁ EXISTE
# ==========================================================

def dataset_ja_dividido(caminho_dividido):
    """
    Verifica se já existem pastas train/val/test com conteúdo.
    """
    splits = ['train', 'val', 'test']

    if not os.path.exists(caminho_dividido):
        return False

    for split in splits:
        split_path = os.path.join(caminho_dividido, split)
        if not os.path.exists(split_path) or len(os.listdir(split_path)) == 0:
            return False

    return True


# ==========================================================
# 3️⃣ FUNÇÃO DE SPLIT
# ==========================================================

def organizar_dataset_milho(
    caminho_dados_originais,
    caminho_dados_divididos,
    test_size=0.3,
    random_state=42
):
    """
    Divide o dataset em train (70%), val (15%) e test (15%)
    """

    if dataset_ja_dividido(caminho_dados_divididos):
        print("Dataset já organizado. Pulando etapa de split.")
        return

    print("Organizando dataset...")

    # Remove pasta antiga se existir
    if os.path.exists(caminho_dados_divididos):
        shutil.rmtree(caminho_dados_divididos)

    os.makedirs(caminho_dados_divididos)

    # Criar estrutura de pastas
    for split in ['train', 'val', 'test']:
        for nome_classe in os.listdir(caminho_dados_originais):
            os.makedirs(
                os.path.join(caminho_dados_divididos, split, nome_classe),
                exist_ok=True
            )

    # Realiza o split por classe
    for nome_classe in os.listdir(caminho_dados_originais):

        caminho_classe = os.path.join(caminho_dados_originais, nome_classe)

        if not os.path.isdir(caminho_classe):
            continue

        imagens = os.listdir(caminho_classe)

        imagens_treino, imagens_temp = train_test_split(
            imagens,
            test_size=test_size,
            random_state=random_state
        )

        imagens_val, imagens_teste = train_test_split(
            imagens_temp,
            test_size=0.5,
            random_state=random_state
        )

        # Função interna de cópia
        def copiar(lista, split):
            for img in lista:
                origem = os.path.join(caminho_classe, img)
                destino = os.path.join(
                    caminho_dados_divididos,
                    split,
                    nome_classe,
                    img
                )
                shutil.copy(origem, destino)

        copiar(imagens_treino, "train")
        copiar(imagens_val, "val")
        copiar(imagens_teste, "test")

    print("Dataset organizado com sucesso!")


# ==========================================================
# 4️⃣ TRANSFORMS (DATA AUGMENTATION)
# ==========================================================

def get_transforms_milho():

    transformacoes_treino = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transformacoes_teste_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transformacoes_treino, transformacoes_teste_val


# ==========================================================
# 5️⃣ DATALOADERS
# ==========================================================

def criar_dataloaders_milho(caminho_dados_divididos, batch_size=32):

    try:
        # Verifica se diretórios existem
        for split in ["train", "val", "test"]:
            caminho_split = os.path.join(caminho_dados_divididos, split)
            if not os.path.exists(caminho_split):
                raise FileNotFoundError(f"Pasta não encontrada: {caminho_split}")

        transform_treino, transform_teste_val = get_transforms_milho()

        dados_treino = datasets.ImageFolder(
            root=os.path.join(caminho_dados_divididos, "train"),
            transform=transform_treino
        )

        dados_validacao = datasets.ImageFolder(
            root=os.path.join(caminho_dados_divididos, "val"),
            transform=transform_teste_val
        )

        dados_teste = datasets.ImageFolder(
            root=os.path.join(caminho_dados_divididos, "test"),
            transform=transform_teste_val
        )

        # Verifica se existem imagens
        if len(dados_treino) == 0:
            raise ValueError("Dataset de treino está vazio.")
        if len(dados_validacao) == 0:
            raise ValueError("Dataset de validação está vazio.")
        if len(dados_teste) == 0:
            raise ValueError("Dataset de teste está vazio.")

        loader_treino = DataLoader(
            dados_treino,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        loader_validacao = DataLoader(
            dados_validacao,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        loader_teste = DataLoader(
            dados_teste,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print("DataLoaders criados com sucesso!")

        return loader_treino, loader_validacao, loader_teste

    except Exception as e:
        print(f"Erro ao criar DataLoaders: {e}")
        return None, None, None