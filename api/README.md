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

## Rodando localmente

Na raiz de `api/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Deploy no Render (monorepo)

No Render, configure o serviço Python apontando para a pasta `api` como Root Directory
e use o start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```



## Posso usar só o código da pasta `src`?

Curto: **não para inferência em produção**.

- A pasta `src` define arquitetura e pipeline, mas **não substitui o modelo treinado**.
- Para a API prever corretamente, você precisa de um artefato com os pesos (checkpoint/MLflow model).
- Nesta API, o carregamento espera um diretório MLflow válido (com arquivo `MLmodel`).

Nesta API, o caminho do modelo é fixo e simples:

- `api/models/modelo_milho_doencas_pyfunc`

Isso ajuda a evitar confusão e facilita manutenção para quem está começando.

