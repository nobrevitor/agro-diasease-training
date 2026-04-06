# Agro Disease Detection API

## Objetivo: 

O projeto tem como objetivo desenvolver uma solução de deep learning capaz de identificar doenças agrícolas em grãos de diferentes culturas. A aplicação permite compreender com precisão qual doença está afetando a plantação, apoiando a tomada de decisões mais inteligentes no manejo agrícola.

Com a identificação correta da doença, torna-se possível utilizar defensivos agrícolas de forma direcionada e eficiente, reduzindo o uso excessivo desses insumos. Essa abordagem contribui para a diminuição dos custos de produção, ao mesmo tempo em que minimiza os impactos ambientais causados pelo uso indiscriminado de defensivos.

Dessa forma, o projeto promove ganhos econômicos para os agricultores e reforça práticas agrícolas mais sustentáveis, aliando produtividade à preservação do meio ambiente.

Este projeto implementa uma arquitetura de MLOps para detecção de doenças agrícolas utilizando modelos de Redes Neurais Convolucionais (CNNs) especializados por cultura.

Atualmente, o sistema suporta:

🌽 Milho

🌱 Soja

## Arquitetura do Projeto:

O projeto foi dividido em três camadas principais:

#### 1. Treinamento e Versionamento:

- Realizado no Databricks Free Edition
- Utilização do MLflow para rastreamento de experimentos e métricas
- Seleção do melhor modelo por desempenho
- Exportação do melhor modelo e versionamento do artefato dentro deste repositório para inferência local

#### 2. Serviço de Inferência:

- API construída com FastAPI
- Deploy realizado no Render
- Carregamento do modelo a partir do diretório local (`MODEL_MILHO_PATH`, padrão: `models/modelo_milho_doencas_pyfunc`)
- Endpoints REST para inferência em tempo real

##### Onde salvar o modelo local?

- **Não é obrigatório salvar dentro de `app/` ou `functions/`**.
- O caminho padrão esperado é `models/modelo_milho_doencas_pyfunc` (na raiz do projeto).
- Você pode salvar em qualquer pasta e configurar via variável de ambiente:

```bash
MODEL_MILHO_PATH=models/meu_modelo
```

- O diretório precisa ser um artefato MLflow válido (com arquivo `MLmodel`).

#### 3. Interface de Usuário

- Aplicação Streamlit consumindo a API
- Upload de imagens e visualização dos resultados

## Endpoints disponíveis: 

Milho
POST /predict/milho

Soja
POST /predict/soja


Input: imagem da folha
Output: classe predita da doença

## Tecnologias Utilizadas:

- Python
- PyTorch
- FastAPI
- MLflow
- Databricks
- Render
- Streamlit
