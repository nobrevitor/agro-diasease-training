# 🌱 Agro Disease Classification

Projeto de **Deep Learning aplicado à agricultura** com foco na identificação automática de **doenças e pragas em culturas agrícolas**, utilizando imagens de folhas e plantas.

Atualmente, o projeto contempla modelos independentes para **milho** e **soja**, permitindo escalabilidade, manutenção simples e experimentação controlada via **MLflow**.

---

## Objetivo

Desenvolver uma solução baseada em **Redes Neurais Convolucionais (CNNs)** capaz de identificar doenças agrícolas a partir de imagens, apoiando a **tomada de decisão inteligente no campo**.

Com isso, o projeto busca:

* Reduzir o uso indiscriminado de defensivos agrícolas
* Diminuir custos operacionais para produtores
* Minimizar impactos ambientais
* Aumentar a eficiência no manejo das culturas

---

## Abordagem Técnica

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
├── api/
|   ├── app/
|   |   └── main.py
|   |
|   ├── functions/
|   |   ├── model.py
|   |   ├── preprocessing.py
|   |   └── schema.py
|   |
|   ├── models/
|   |   ├── modelo_milho_doencas_pyfunc/
|   |   |   └── MLmodel
|   |   └── modelo_soja_doencas_pyfunc/
|   |       └── MLmodel
|   |
|   ├── src/
|   |   └── models/
|   |       └── milho_models.py
|   |
|   ├── requirements.txt
|   └── Dockerfile
|    
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
└── src/
    ├── dataset/
    │   ├── milho_dataset.py
    │   └── soja_dataset.py
    |
    ├── models/
    │   ├── milho_model.py
    │   └── soja_model.py
    │
    ├── preprocessing/
    │   ├── milho_preprocessing.py
    │   └── soja_preprocessing.py
    │
    ├── train.py
    └── evaluate.py
```

---

## Tratamento de Overfitting (Modelo de Soja)

Durante a fase de validação do modelo de soja, identificamos um cenário de overfitting (sobreajuste). O problema foi detectado ao submeter o modelo a um dataset de teste externo, previamente limpo e normalizado, onde a performance foi significativamente inferior aos dados de treino.Para mitigar esse problema e melhorar a generalização do modelo, aplicamos as seguintes estratégias:

Ajuste de Hiperparâmetros: Refinamos a taxa de aprendizado (LR ou Learning Rate) para permitir uma convergência mais estável e evitar que o modelo ficasse "preso" em mínimos locais de ruído dos dados de treino. 

Data Augmentation Estratégico: 
    Classes Minoritárias: Aplicamos um aumento agressivo de dados (rotações, flips, ajustes de brilho e contraste) para equilibrar a representatividade dessas classes.
    Classes Majoritárias: Reduzimos drasticamente o volume de dados e a intensidade das transformações para evitar que o modelo se tornasse tendencioso (bias) para as classes com mais amostras.
    Validação Cruzada: O loop de teste foi reestruturado para garantir que a normalização dos dados de produção fosse idêntica à do treinamento.

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
* **Streamlit** → Interface para upload de imagens

Fluxo previsto:

```text
Usuário → Streamlit → FastAPI → Modelo → Predição
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
* Streamlit

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
