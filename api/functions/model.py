import logging
from pathlib import Path

import mlflow.pyfunc

logger = logging.getLogger(__name__)

_model_milho = None
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "modelo_milho_doencas_pyfunc"


def _load_model_milho():
    if not _MODEL_DIR.exists():
        raise RuntimeError(f"Modelo não encontrado em: {_MODEL_DIR}")

    if not (_MODEL_DIR / "MLmodel").exists():
        raise RuntimeError(
            f"A pasta do modelo não é um artefato MLflow válido: {_MODEL_DIR}"
        )

    try:
        return mlflow.pyfunc.load_model(_MODEL_DIR.as_uri())
    except Exception as exc:
        raise RuntimeError(f"Erro ao carregar modelo de milho: {exc}") from exc


def get_model(nome_modelo: str):
    global _model_milho

    if nome_modelo != "milho":
        raise RuntimeError(f"Modelo '{nome_modelo}' não encontrado")

    if _model_milho is None:
        _model_milho = _load_model_milho()
        logger.info("Modelo de milho carregado com sucesso de '%s'.", _MODEL_DIR)

    return _model_milho
