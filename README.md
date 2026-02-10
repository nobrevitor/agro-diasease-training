# 🌱 Agro Disease Classification

Projeto de **Deep Learning aplicado à agricultura** com foco na identificação automática de **doenças e pragas em culturas agrícolas**, utilizando imagens de folhas e plantas.

Atualmente, o projeto contempla modelos independentes para **milho** e **soja**, permitindo escalabilidade, manutenção simples e experimentação controlada via **MLflow**.

---

## 🎯 Objetivo

Desenvolver uma solução baseada em **Redes Neurais Convolucionais (CNNs)** capaz de identificar doenças agrícolas a partir de imagens, apoiando a **tomada de decisão inteligente no campo**.

Com isso, o projeto busca:

* Reduzir o uso indiscriminado de defensivos agrícolas
* Diminuir custos operacionais para produtores
* Minimizar impactos ambientais
* Aumentar a eficiência no manejo das culturas

---

## 🧠 Abordagem Técnica

* Modelos independentes por cultura (ex: milho e soja)
* Treinamento supervisionado com imagens rotuladas
* Arquiteturas CNN modernas (ResNet, EfficientNet, etc.)
* Experimentos rastreados com **MLflow**
* Pipeline preparado para produção via **FastAPI + Streamlit**

---

## 🗂️ Estrutura do Repositório

```bash
agro-disease-classification/
│
├── README.md
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Organização e preparação do dataset
│   ├── 02_training_mlflow.ipynb        # Treinamento e experimentação com MLflow
│   └── 03_evaluation.ipynb             # Avaliação final dos modelos
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Parâmetros globais e paths
│   ├── preprocessing.py               # Transformações e data augmentation
│   ├── dataset.py                     # Dataset e DataLoader
│   ├── model.py                       # Arquiteturas de modelos
│   ├── train.py                       # Loop de treino (agnóstico a experimento)
│   ├── evaluate.py                    # Métricas e avaliação
│   └── inference.py                   # Inferência para produção
│
├── mlflow/
│   └── README.md                      # Organização dos experimentos
│
├── scripts/
│   ├── register_model.py              # Registro de modelo vencedor
│   └── batch_inference.py             # Inferência em lote
│
└── docs/
    ├── architecture.md                # Arquitetura do projeto
    └── experiment_tracking.md         # Estratégia de experimentos
```

---

## 🔬 Experimentos e MLflow

Os experimentos são definidos **fora da lógica de treino**, permitindo:

* Comparação justa entre modelos
* Reprodução de resultados
* Seleção automática do melhor modelo

Cada experimento registra:

* Hiperparâmetros
* Métricas de treino e validação
* Artefatos do modelo

---

## 🚀 Deploy (Roadmap)

O projeto está preparado para produção utilizando:

* **FastAPI** → Servir o modelo como API REST
* **Render** → Hospedagem gratuita do backend
* **React** → Interface para upload de imagens

Fluxo previsto:

```text
Usuário → React → FastAPI → Modelo → Predição
```

---

## 🛠️ Tecnologias Utilizadas

* Python
* PyTorch
* Torchvision
* Scikit-learn
* MLflow
* Databricks (Free Edition)
* FastAPI
* React

---

## 👥 Colaboradores

* **Vitor Nobre** – Data Scientist / ML Engineer
* **Jefferson** – Cientista da Computação

---

## 📌 Observações

* Os dados utilizados **não estão versionados** neste repositório
* O projeto segue boas práticas de MLOps e versionamento de código
* Estrutura pensada para escalar para novas culturas agrícolas

---

🌾 *Tecnologia aplicada para uma agricultura mais inteligente e sustentável.*
