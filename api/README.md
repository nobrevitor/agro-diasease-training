# Agro Disease API (FastAPI)

API de inferência para classificação de doenças em folhas de milho.

## Estrutura da pasta `api`

```text
api/
├── app/
│   └── main.py                  # Endpoints FastAPI
├── functions/
│   ├── model.py                 # Carregamento do modelo MLflow local
│   ├── preprocessing.py         # Pré-processamento da imagem
│   └── schema.py                # Schemas de resposta
├── src/
|   └── models/
|       └── milho_models.py      # Móudlo de treinamento do modelo salvo
├── models/
│   └── modelo_milho_doencas_pyfunc/
│       └── MLmodel              # Artefato MLflow versionado no repo
├── requirements.txt
└── Dockerfile
```

## Endpoints

- `GET /health` → status da API
- `GET /health?check_model=true` → status + valida carregamento do modelo
- `POST /predict` → predição a partir de upload de imagem

### Exemplo de uso (`/predict`)

```bash
curl -X POST \
  -F "file=@folha.jpg" \
  http://localhost:8000/predict
```

Resposta esperada:

```json
{
  "prediction": "Common Rust",
  "confidence": 0.97
}
```

