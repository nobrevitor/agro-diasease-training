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

```
agro-disease-classification/
│
├── README.md
│
├── notebooks/
│   ├── milho/
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_training_mlflow.ipynb
│   │   └── 03_evaluation.ipynb
│   │
│   └── soja/
│       ├── 01_data_preparation.ipynb
│       ├── 02_training_mlflow.ipynb
│       └── 03_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── milho_model.py
│   │   └── soja_model.py
│   │
│   ├── preprocessing/
│   │   ├── milho_preprocessing.py
│   │   └── soja_preprocessing.py
│   │
│   ├── dataset/
│   │   ├── milho_dataset.py
│   │   └── soja_dataset.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
└── docs/
    ├── architecture.md
    └── experiment_tracking.md
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
