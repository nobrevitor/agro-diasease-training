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

#import logging
#import os
#from pathlib import Path

#import mlflow.pyfunc

#logger = logging.getLogger(__name__)

#model_milho = None

#DEFAULT_MODEL_PATH = "models/modelo_milho_doencas_pyfunc"


#def _resolve_model_path() -> Path:
#    model_path = os.getenv("MODEL_MILHO_PATH", DEFAULT_MODEL_PATH).strip()
#    if not model_path:
#        raise RuntimeError("MODEL_MILHO_PATH está vazio.")

#    path = Path(model_path)
#    if not path.is_absolute():
#        path = Path(__file__).resolve().parent.parent / path

#    if not path.exists():
#        raise RuntimeError(
#            f"Modelo local não encontrado em '{path}'. "
#            "Salve o artefato no repositório ou ajuste MODEL_MILHO_PATH."
#        )

#    if path.is_dir() and not (path / "MLmodel").exists():
#        raise RuntimeError(
#            f"O diretório '{path}' não parece ser um modelo MLflow válido "
#            "(arquivo 'MLmodel' não encontrado)."
#        )

#    return path



def get_model(nome_modelo: str):
    global _model_milho

    if nome_modelo != "milho":
        raise RuntimeError(f"Modelo '{nome_modelo}' não encontrado")

    if _model_milho is None:
        _model_milho = _load_model_milho()
        logger.info("Modelo de milho carregado com sucesso de '%s'.", _MODEL_DIR)

    return _model_milho
