# CP5 — Predição de Evasão Universitária

Projeto desenvolvido para a disciplina de **Data Science & Statistical Computing**, da FIAP, em 2026.

## Links do projeto

- **Google Colab:** https://colab.research.google.com/drive/1ylYpyKKmV3L_nNixySYv4XpBLByNVtl2?usp=sharing
- **Aplicação Streamlit:** https://cp5datascience2026-2neg4vhdjyuavsdjaibctn.streamlit.app/

## Integrantes

| Nome | RM |
|---|---|
| Felipe Viana | 565341 |
| Felipe Bonilha | 562356 |
| Joan Ferreira | 562913 |
| Levi de Jesus | 563279 |
| Luigi Borghi | 563096 |

## 1. Objetivo

Investigar se informações acadêmicas e de engajamento podem ser utilizadas para identificar estudantes com padrões associados à evasão universitária.

A proposta consiste em desenvolver uma ferramenta experimental de classificação que possa apoiar o acompanhamento acadêmico, sem substituir a avaliação humana.

## 2. Base de dados

Foi utilizada a **StudentDropoutDataset**, disponibilizada pela Universitat Politècnica de València (UPV), na plataforma Zenodo.

Foram analisados registros dos anos de 2018, 2021 e 2022.

Após o tratamento e agregação por estudante/ano, a base apresentou:

- **60.924 observações** de aluno/ano;
- **39.364 estudantes únicos**;
- **7,05% de registros classificados como evasão**.

As variáveis incluem informações acadêmicas, desempenho anterior, créditos matriculados e indicadores de utilização da plataforma de ensino.

## 3. Modelos avaliados

Foram utilizados três algoritmos de aprendizado supervisionado:

- Random Forest;
- XGBoost;
- LightGBM.

Para cada algoritmo, foram realizadas três etapas:

1. Treinamento de um modelo inicial (baseline).
2. Otimização de hiperparâmetros com Grid Search.
3. Otimização de hiperparâmetros com Optuna.

A avaliação utilizou validação cruzada com cinco folds, mantendo o conjunto de teste separado durante a seleção e otimização dos modelos.

## 4. Modelo selecionado

O modelo escolhido foi o **XGBoost otimizado com Optuna**, que apresentou o melhor F1-score médio na validação cruzada.

| Modelo otimizado com Optuna | F1 Validação |
|---|---:|
| Random Forest | 0,3202 |
| XGBoost | **0,3457** |
| LightGBM | 0,3127 |

Após a seleção, foi definido um limiar de classificação de **0,34**, utilizando dados de treinamento e validação e priorizando o Recall por meio do F2-score.

## 5. Resultados finais

No conjunto de teste, foram obtidos os seguintes resultados:

| Métrica | Resultado |
|---|---:|
| Recall | 66,3% |
| Precision | 20,4% |
| F1-score | 0,312 |
| F2-score | 0,457 |
| ROC-AUC | 0,822 |
| PR-AUC | 0,310 |

Das **834 evasões reais** presentes no conjunto de teste:

- 553 foram identificadas corretamente;
- 281 não foram identificadas;
- 2.157 falsos alertas foram gerados.

O modelo conseguiu identificar uma parcela relevante dos casos reais de evasão, mas apresentou um número elevado de falsos positivos.

## 6. Aplicação Streamlit

A aplicação foi desenvolvida para apresentar os resultados do projeto e demonstrar o funcionamento do modelo selecionado.

Possui três áreas principais:

- **Visão Geral:** apresentação do problema, objetivo e base utilizada.
- **Modelos:** comparação dos algoritmos, otimizações e resultados.
- **Análise de Caso:** demonstração da classificação de um estudante real do conjunto de teste.

O Streamlit utiliza os artefatos do modelo XGBoost e do pré-processamento exportados a partir do notebook.

O projeto também inclui um procedimento de comparação entre as previsões do notebook e da aplicação.

## 7. Estrutura do repositório

```text
CP5_DataScience_2026/
│
├── notebook/
│   └── CP5_DataScience.ipynb
│
├── modelo/
│   ├── config.json
│   ├── modelo_evasao_xgboost.pkl
│   ├── preprocessador.pkl
│   └── xgboost_modelo.json
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Como executar

### Instalar as dependências

```bash
pip install -r requirements.txt
```

### Executar a aplicação Streamlit

```bash
streamlit run app.py
```

A aplicação será iniciada localmente e poderá ser acessada pelo navegador.

### Executar o notebook

O notebook está disponível na pasta `notebook/` ou pelo link do Google Colab apresentado no início deste documento.

Para reproduzir a análise, é necessário disponibilizar os dados originais da pesquisa, conforme as instruções de carregamento do notebook, e executar as células na ordem apresentada.

## 9. Limitações

O projeto apresenta limitações importantes:

- A base possui classes desbalanceadas, com aproximadamente 7% de registros de evasão.
- O modelo apresenta quantidade elevada de falsos alertas.
- Algumas informações de engajamento estão ausentes nos dados originais.
- Os resultados não comprovam, por si só, a capacidade de antecipar a evasão em futuras turmas.
- A aplicação não deve ser utilizada para decisões automáticas sobre estudantes.

Portanto, o sistema deve ser interpretado como um **protótipo acadêmico de classificação e apoio à análise**, que necessita de validação adicional para utilização institucional.

## 10. Conclusão

O projeto permitiu comparar Random Forest, XGBoost e LightGBM sob um mesmo protocolo experimental, incluindo treinamento inicial, Grid Search, Optuna e avaliação final.

O XGBoost otimizado com Optuna apresentou o melhor desempenho na validação cruzada e foi selecionado para integração com o Streamlit.

Os resultados demonstram que é possível identificar padrões associados à evasão universitária, mas também evidenciam limitações relacionadas à precisão dos alertas e à generalização temporal.

O trabalho representa uma aplicação prática de Machine Learning para análise de dados educacionais, com possibilidades de aprimoramento em estudos futuros.
