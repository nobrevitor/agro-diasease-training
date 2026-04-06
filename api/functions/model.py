import logging
import os
from pathlib import Path

import mlflow.pyfunc

logger = logging.getLogger(__name__)

model_milho = None

DEFAULT_MODEL_PATH = "models/modelo_milho_doencas_pyfunc"

def _resolve_model_path() -> Path:
    model_path = os.getenv("MODEL_MILHO_PATH", DEFAULT_MODEL_PATH).strip()
    if not model_path:
        raise RuntimeError("MODEL_MILHO_PATH está vazio.")

    path = Path(model_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    if not path.exists():
        raise RuntimeError(
            f"Modelo local não encontrado em '{path}'. "
            "Salve o artefato no repositório ou ajuste MODEL_MILHO_PATH."
        )

    if path.is_dir() and not (path / "MLmodel").exists():
        raise RuntimeError(
            f"O diretório '{path}' não parece ser um modelo MLflow válido "
            "(arquivo 'MLmodel' não encontrado)."
        )

    return path


def get_model(nome_modelo):
    global model_milho

    if nome_modelo != "milho":
        raise RuntimeError(f"Modelo '{nome_modelo}' não encontrado")

    if model_milho is None:
        try:
            model_path = _resolve_model_path()
            model_milho = mlflow.pyfunc.load_model(model_path.as_uri())
            logger.info("Modelo de milho carregado com sucesso de '%s'.", model_path)
        except Exception as exc:
            raise RuntimeError(f"Falha ao carregar modelo local: {exc}") from exc

    return model_milho
