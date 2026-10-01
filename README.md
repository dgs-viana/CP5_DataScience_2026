# CP5 — Predição de Evasão Universitária

Projeto desenvolvido para a disciplina de Data Science & Statistical Computing
da FIAP.

#Link Colab e Streamlit 

https://colab.research.google.com/drive/1LSHjRN5aKHVYvdugIyWAaTEqFrQgYwGm?usp=sharing

https://cp5datascience2026-2neg4vhdjyuavsdjaibctn.streamlit.app/

# Alunos 

Felipe Viana - RM 565341 

Felipe Bonilha - RM 562356 

Joan Ferreira - RM 562913 

Levi de Jesus - RM 563279 

Luigi Borghi - RM 563096

## Objetivo

Investigar se informações acadêmicas e de engajamento podem ser utilizadas para
identificar estudantes com padrões associados à evasão universitária.

A solução foi desenvolvida como uma ferramenta experimental de triagem e apoio
ao acompanhamento acadêmico, e não como mecanismo automático de decisão.

## Base de dados

Foi utilizada a StudentDropoutDataset, disponibilizada pela Universitat
Politècnica de València (UPV) no Zenodo.

Dados utilizados dos anos de:

- 2018
- 2021
- 2022

Após o tratamento e agregação por estudante/ano, a base utilizada na modelagem
possui 60.924 observações e 39.364 estudantes únicos.

A classe de evasão representa aproximadamente 7,05% das observações.

## Modelos avaliados

Foram comparados:

- Random Forest
- XGBoost
- LightGBM

Para cada algoritmo foram avaliadas:

- versão baseline;
- otimização com Grid Search;
- otimização com Optuna.

O protocolo experimental utilizou validação cruzada sobre o conjunto de treino,
mantendo o conjunto de teste isolado até a avaliação final.

## Modelo selecionado

O modelo selecionado foi o XGBoost otimizado com Optuna.

F1 médio na validação cruzada:

- XGBoost Optuna: 0,3457

Após a escolha do modelo, o limiar operacional foi ajustado utilizando somente
treino e validação, priorizando Recall por meio do F2-Score.

Limiar final:

- 0,34

## Resultado final

No conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Recall | 66,3% |
| Precision | 20,4% |
| F1-Score | 0,312 |
| F2-Score | 0,457 |
| ROC-AUC | 0,822 |
| PR-AUC | 0,310 |

Das 834 evasões reais presentes no conjunto de teste:

- 553 foram identificadas;
- 281 não foram identificadas;
- 2.157 falsos alertas foram gerados.

Os resultados indicam que o modelo pode ser utilizado como ferramenta de
priorização para acompanhamento acadêmico, mas não possui desempenho adequado
para decisões automáticas sobre estudantes.

## Streamlit

A aplicação possui três áreas:

- Visão Geral do projeto;
- comparação dos modelos e resultados;
- análise de um caso real do conjunto de teste.

O Streamlit utiliza o mesmo pré-processamento e o mesmo modelo XGBoost final do
notebook.

Também foi realizado um teste de paridade entre notebook e aplicação.

## Executando localmente

Instale as dependências:

```bash
pip install -r requirements.txt
