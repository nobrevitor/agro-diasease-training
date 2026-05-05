"""
soja_preprocessing.py

Módulo de pré-processamento para o projeto de visão computacional com soja.

Este arquivo foi criado para ser importado em notebooks Databricks.
"""

import random
import shutil
from pathlib import Path

import kagglehub
from PIL import Image, ImageEnhance


SEED = 42
random.seed(SEED)

BASE_DIR = Path("/Volumes/workspace/agro-diasease/soja")

DATASET_ORGANIZADO = BASE_DIR / "dataset_soja"
DATASET_SPLIT = BASE_DIR / "dataset_soja_split"

PROPORCAO_TREINO = 0.70
PROPORCAO_VAL = 0.15
PROPORCAO_TESTE = 0.15

CLASSE_SAUDAVEL = "Soja_Saudavel"

MAX_IMAGENS_SAUDAVEL_TREINO = 100
ALVO_AUGMENTACAO_DOENTES = 400

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


MAPA_CLASSES_DOENTES = {
    "ferrugen": "Ferrugem",
    "Yellow Mosaic": "Mosaico_Amarelo",
    "brown_spot": "Mancha_Parda",
    "powdery_mildew": "Oidio",
    "septoria": "Septoriose",
    "Southern blight": "Podridao_Sul",
    "Mossaic Virus": "Virus_Mosaico",
    "Sudden Death Syndrone": "Sindrome_Morte_Subita",
    "bacterial_blight": "Queima_Bacteriana",
    "crestamento": "Crestamento",
}

# FUNÇÕES AUXILIARES


def criar_pasta(caminho: Path) -> None:
    caminho.mkdir(parents=True, exist_ok=True)


def limpar_pasta(caminho: Path) -> None:
    if caminho.exists():
        shutil.rmtree(caminho)
    criar_pasta(caminho)


def listar_imagens(pasta: Path) -> list[Path]:
    if not pasta.exists():
        return []

    return [
        arquivo
        for arquivo in pasta.iterdir()
        if arquivo.is_file()
        and arquivo.suffix.lower() in EXTENSOES_VALIDAS
    ]


def copiar_imagem(origem: Path, destino: Path) -> None:
    """
    Copia uma imagem evitando sobrescrever arquivos com o mesmo nome.
    """

    criar_pasta(destino.parent)

    destino_final = destino
    contador = 1

    while destino_final.exists():
        destino_final = destino.with_name(
            f"{destino.stem}_{contador}{destino.suffix}"
        )
        contador += 1

    shutil.copy2(origem, destino_final)

# DOWNLOAD DOS DATASETS

def baixar_datasets() -> tuple[Path, Path]:
    """
    Baixa os datasets necessários via KaggleHub.
    """

    print("Baixando dataset de soja doente...")
    pasta_soja_doente = Path(
        kagglehub.dataset_download(
            "sivm205/soybean-diseased-leaf-dataset"
        )
    )

    print("Baixando dataset PlantVillage...")
    pasta_plantvillage = Path(
        kagglehub.dataset_download(
            "mohitsingh1804/plantvillage"
        )
    )

    print("\nDownloads finalizados:")
    print(f"Soja doente: {pasta_soja_doente}")
    print(f"PlantVillage: {pasta_plantvillage}")

    return pasta_soja_doente, pasta_plantvillage


# INSPEÇÃO DOS DATASETS ORIGINAIS

def listar_pastas_dataset(caminho_dataset: Path) -> None:
    caminho_dataset = Path(caminho_dataset)

    if not caminho_dataset.exists():
        print(f"Caminho não encontrado: {caminho_dataset}")
        return

    print(f"\nPastas encontradas em {caminho_dataset}:")
    for item in sorted(caminho_dataset.iterdir()):
        if item.is_dir():
            print(item.name)


def contar_imagens_por_classe(caminho_dataset: Path) -> None:
    caminho_dataset = Path(caminho_dataset)

    if not caminho_dataset.exists():
        print(f"Caminho não encontrado: {caminho_dataset}")
        return

    print(f"\nContagem de imagens em {caminho_dataset}:")

    for classe_dir in sorted(caminho_dataset.iterdir()):
        if classe_dir.is_dir():
            qtd = len(listar_imagens(classe_dir))
            print(f"{classe_dir.name}: {qtd}")


# ORGANIZAÇÃO DO DATASET DE SOJA

def criar_estrutura_dataset_organizado(
    pasta_destino: Path = DATASET_ORGANIZADO
) -> None:
    pasta_destino = Path(pasta_destino)

    criar_pasta(pasta_destino / CLASSE_SAUDAVEL)

    for classe_pt in MAPA_CLASSES_DOENTES.values():
        criar_pasta(pasta_destino / classe_pt)


def copiar_soja_saudavel(
    pasta_plantvillage: Path,
    pasta_destino: Path = DATASET_ORGANIZADO
) -> None:
    pasta_plantvillage = Path(pasta_plantvillage)
    pasta_destino = Path(pasta_destino)

    origens = [
        pasta_plantvillage / "PlantVillage" / "train" / "Soybean___healthy",
        pasta_plantvillage / "PlantVillage" / "val" / "Soybean___healthy",
    ]

    destino = pasta_destino / CLASSE_SAUDAVEL
    total = 0

    for origem in origens:
        imagens = listar_imagens(origem)

        if not imagens:
            print(f"Aviso: pasta saudável não encontrada ou vazia: {origem}")
            continue

        for imagem in imagens:
            copiar_imagem(imagem, destino / imagem.name)
            total += 1

    print(f"Soja saudável copiada: {total} imagens")


def copiar_soja_doente(
    pasta_soja_doente: Path,
    pasta_destino: Path = DATASET_ORGANIZADO
) -> None:
    pasta_soja_doente = Path(pasta_soja_doente)
    pasta_destino = Path(pasta_destino)

    for classe_original, classe_pt in MAPA_CLASSES_DOENTES.items():
        origem = pasta_soja_doente / classe_original
        destino = pasta_destino / classe_pt

        imagens = listar_imagens(origem)

        if not imagens:
            print(f"Aviso: classe não encontrada ou vazia: {origem}")
            continue

        for imagem in imagens:
            copiar_imagem(imagem, destino / imagem.name)

        print(f"{classe_original} -> {classe_pt}: {len(imagens)} imagens")


def organizar_dataset_soja(
    pasta_soja_doente: Path,
    pasta_plantvillage: Path,
    pasta_destino: Path = DATASET_ORGANIZADO,
    limpar_destino: bool = True
) -> None:
    pasta_destino = Path(pasta_destino)

    print("\nOrganizando dataset de soja...")

    if limpar_destino:
        limpar_pasta(pasta_destino)
    else:
        criar_pasta(pasta_destino)

    criar_estrutura_dataset_organizado(pasta_destino)
    copiar_soja_saudavel(pasta_plantvillage, pasta_destino)
    copiar_soja_doente(pasta_soja_doente, pasta_destino)

    print("\nDataset organizado criado em:")
    print(pasta_destino)


# SPLIT TRAIN / VAL / TEST

def dividir_lista_imagens(
    imagens: list[Path],
    proporcao_treino: float = PROPORCAO_TREINO,
    proporcao_val: float = PROPORCAO_VAL
) -> tuple[list[Path], list[Path], list[Path]]:

    imagens = imagens.copy()
    random.shuffle(imagens)

    total = len(imagens)

    if total == 0:
        return [], [], []

    n_train = int(total * proporcao_treino)
    n_val = int(total * proporcao_val)

    if total >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)

    imagens_train = imagens[:n_train]
    imagens_val = imagens[n_train:n_train + n_val]
    imagens_test = imagens[n_train + n_val:]

    if total >= 3 and len(imagens_test) == 0:
        imagens_test = [imagens_train.pop()]

    return imagens_train, imagens_val, imagens_test


def criar_split_dataset(
    pasta_origem: Path = DATASET_ORGANIZADO,
    pasta_destino: Path = DATASET_SPLIT,
    limpar_destino: bool = True
) -> None:
    pasta_origem = Path(pasta_origem)
    pasta_destino = Path(pasta_destino)

    print("\nCriando split train/val/test...")

    if limpar_destino:
        limpar_pasta(pasta_destino)
    else:
        criar_pasta(pasta_destino)

    for split in ["train", "val", "test"]:
        criar_pasta(pasta_destino / split)

    classes = [
        pasta
        for pasta in pasta_origem.iterdir()
        if pasta.is_dir()
    ]

    for pasta_classe in classes:
        classe = pasta_classe.name
        imagens = listar_imagens(pasta_classe)

        imagens_train, imagens_val, imagens_test = dividir_lista_imagens(imagens)

        split_map = {
            "train": imagens_train,
            "val": imagens_val,
            "test": imagens_test,
        }

        for split, imagens_split in split_map.items():
            destino_classe = pasta_destino / split / classe
            criar_pasta(destino_classe)

            for imagem in imagens_split:
                copiar_imagem(imagem, destino_classe / imagem.name)

        print(
            f"{classe}: "
            f"train={len(imagens_train)}, "
            f"val={len(imagens_val)}, "
            f"test={len(imagens_test)}"
        )

    print("\nDataset com split criado em:")
    print(pasta_destino)


# BALANCEAMENTO E AUGMENTAÇÃO OFFLINE

def limitar_classe_saudavel_treino(
    pasta_split: Path = DATASET_SPLIT,
    max_imagens: int = MAX_IMAGENS_SAUDAVEL_TREINO
) -> None:
    pasta_split = Path(pasta_split)
    pasta_saudavel = pasta_split / "train" / CLASSE_SAUDAVEL

    imagens = listar_imagens(pasta_saudavel)

    if len(imagens) <= max_imagens:
        print(
            f"\nClasse saudável possui {len(imagens)} imagens. "
            "Nenhum corte necessário."
        )
        return

    random.shuffle(imagens)
    imagens_para_remover = imagens[max_imagens:]

    for imagem in imagens_para_remover:
        imagem.unlink()

    print(
        f"\nClasse saudável reduzida no treino: "
        f"{len(imagens)} -> {max_imagens}"
    )


def augmentar_imagem(imagem: Image.Image) -> Image.Image:
    """
    Aplica augmentação offline simples usando PIL.
    Compatível com Databricks Free Edition.
    """

    angulo = random.uniform(-25, 25)
    imagem_aug = imagem.rotate(angulo)

    if random.random() > 0.5:
        imagem_aug = imagem_aug.transpose(Image.FLIP_LEFT_RIGHT)

    fator_brilho = random.uniform(0.7, 1.3)
    imagem_aug = ImageEnhance.Brightness(imagem_aug).enhance(fator_brilho)

    return imagem_aug


def aplicar_augmentacao_treino(
    pasta_split: Path = DATASET_SPLIT,
    alvo_por_classe: int = ALVO_AUGMENTACAO_DOENTES,
    ignorar_classes: list[str] | None = None
) -> None:
    """
    Aplica augmentação offline nas classes do treino usando PIL.
    Por padrão, ignora a classe saudável.
    """

    if ignorar_classes is None:
        ignorar_classes = [CLASSE_SAUDAVEL]

    pasta_split = Path(pasta_split)
    pasta_treino = pasta_split / "train"

    print("\nAplicando augmentação offline nas classes doentes...")

    for pasta_classe in pasta_treino.iterdir():
        if not pasta_classe.is_dir():
            continue

        classe = pasta_classe.name

        if classe in ignorar_classes:
            continue

        imagens = listar_imagens(pasta_classe)
        qtd_atual = len(imagens)

        print(f"{classe}: {qtd_atual} -> ", end="")

        if qtd_atual >= alvo_por_classe:
            print("ok")
            continue

        if qtd_atual == 0:
            print("sem imagens para augmentar")
            continue

        contador = 0

        while qtd_atual + contador < alvo_por_classe:
            imagem_origem = random.choice(imagens)

            try:
                with Image.open(imagem_origem) as img:
                    img = img.convert("RGB")
                    imagem_aug = augmentar_imagem(img)

                    novo_nome = (
                        f"aug_{contador}_{imagem_origem.stem}"
                        f"{imagem_origem.suffix}"
                    )

                    caminho_saida = pasta_classe / novo_nome
                    imagem_aug.save(caminho_saida)

                    contador += 1

            except Exception as erro:
                print(f"\nErro ao processar {imagem_origem}: {erro}")
                continue

        print(qtd_atual + contador)


# RELATÓRIOS

def mostrar_distribuicao_dataset(
    caminho_dataset: Path = DATASET_ORGANIZADO
) -> None:
    caminho_dataset = Path(caminho_dataset)

    print(f"\nDistribuição de imagens em: {caminho_dataset}")

    if not caminho_dataset.exists():
        print("Caminho não encontrado.")
        return

    for classe_dir in sorted(caminho_dataset.iterdir()):
        if classe_dir.is_dir():
            qtd = len(listar_imagens(classe_dir))
            print(f"{classe_dir.name}: {qtd}")


def mostrar_distribuicao_split(
    pasta_split: Path = DATASET_SPLIT
) -> None:
    pasta_split = Path(pasta_split)

    print("\nDistribuição final por split:")

    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}")

        pasta = pasta_split / split

        if not pasta.exists():
            print("Pasta não encontrada.")
            continue

        for classe_dir in sorted(pasta.iterdir()):
            if classe_dir.is_dir():
                qtd = len(listar_imagens(classe_dir))
                print(f"{classe_dir.name}: {qtd}")


if __name__ == "__main__":
    print(
        "Este arquivo foi criado para ser importado em notebooks.\n\n"
        "Exemplo de uso:\n"
        "from soja_preprocessing import baixar_datasets, organizar_dataset_soja, criar_split_dataset\n\n"
        "Ou, se quiser executar tudo de uma vez:\n"
        "from soja_preprocessing import executar_pipeline_completo\n"
        "executar_pipeline_completo()"
    )