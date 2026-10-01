import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Sistema de Alerta de Evasão",
    page_icon="🎓",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .titulo-projeto {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitulo-projeto {
            font-size: 1.05rem;
            opacity: 0.75;
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARREGAMENTO DOS ARTEFATOS
# ============================================================

PASTA_MODELO = Path("modelo")


@st.cache_resource
def carregar_artefatos():

    preprocessador = joblib.load(
        PASTA_MODELO / "preprocessador.pkl"
    )

    modelo = xgb.XGBClassifier()

    modelo.load_model(
        PASTA_MODELO / "xgboost_modelo.json"
    )

    with open(
        PASTA_MODELO / "config.json",
        "r",
        encoding="utf-8"
    ) as arquivo:

        config = json.load(arquivo)

    return preprocessador, modelo, config


try:

    preprocessador, modelo, config = carregar_artefatos()

except Exception as erro:

    st.error(
        f"Erro ao carregar os arquivos do modelo: {erro}"
    )

    st.stop()


threshold = config["threshold"]
features = config["features"]


# ============================================================
# CASO REAL USADO NO TESTE DE PARIDADE
# ============================================================

aluno_base = {

    "anyo_ingreso": 2014.0,
    "tipo_ingreso": "NAP",

    "nota10_hash": 7.031,
    "nota14_hash": 9.514,

    "estudios_p_hash": "L",
    "estudios_m_hash": "F",

    "dedicacion": "TC",
    "desplazado_hash": "A",

    "preferencia_seleccion": 1.0,

    "anyo_inicio_estudios": 2014,

    "curso_mas_bajo": 1,
    "curso_mas_alto": 4,

    "cred_mat_normal": 42.5,
    "cred_mat_movilidad": 0.0,
    "cred_mat_practicas": 9.5,

    "cred_mat_sem_a": 37.5,
    "cred_mat_sem_b": 18.0,
    "cred_mat_anu": 0.0,

    "cred_mat_total": 55.5,

    "rend_total_ultimo": 79.411765,
    "rend_total_penultimo": 82.5,
    "rend_total_antepenultimo": 77.5,

    "lms_eventos": 180.0,
    "lms_dias_acesso": 0.0,
    "lms_visitas": 0.0,

    "lms_atividades_entregues": 0.0,
    "lms_testes_enviados": 0.0,

    "lms_minutos_totais": 2925.811817,

    "lms_eventos_recursos": 90.0,
    "lms_dias_recursos": 30.0
}


# ============================================================
# FUNÇÃO DE PREVISÃO
# ============================================================

def prever(dados):

    entrada = pd.DataFrame(
        [dados]
    )

    entrada = entrada[
        features
    ]

    entrada_processada = preprocessador.transform(
        entrada
    )

    score = modelo.predict_proba(
        entrada_processada
    )[0, 1]

    previsao = int(
        score >= threshold
    )

    return score, previsao


# ============================================================
# CABEÇALHO
# ============================================================

st.markdown(
    '<p class="titulo-projeto">🎓 Sistema de Alerta de Evasão Universitária</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitulo-projeto">
    Machine Learning aplicado à identificação de estudantes
    com padrões associados à evasão universitária.
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ABAS
# ============================================================

aba_geral, aba_modelos, aba_caso = st.tabs(
    [
        "📊 Visão Geral",
        "🧠 Modelos",
        "🎯 Análise de Caso"
    ]
)


# ============================================================
# ABA 1 - VISÃO GERAL
# ============================================================

with aba_geral:

    st.header(
        "Visão Geral do Projeto"
    )

    st.write(
        """
        O objetivo do projeto é investigar se informações
        acadêmicas e de engajamento podem ser utilizadas para
        identificar estudantes que apresentam padrões associados
        a maior risco de evasão.

        A proposta é utilizar o modelo como ferramenta de
        **triagem e apoio ao acompanhamento acadêmico**.
        """
    )

    st.info(
        """
        O sistema não determina que um estudante irá abandonar
        o curso.

        Ele apenas identifica casos que podem receber prioridade
        para análise e acompanhamento pela instituição.
        """
    )

    st.subheader(
        "Base utilizada"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Observações aluno/ano",
            "60.924"
        )

    with col2:
        st.metric(
            "Estudantes",
            "39.364"
        )

    with col3:
        st.metric(
            "Taxa de evasão",
            "7,05%"
        )

    with col4:
        st.metric(
            "Bases",
            "2018 • 2021 • 2022"
        )

    st.divider()

    col1, col2 = st.columns(
        [1.15, 1]
    )

    with col1:

        st.subheader(
            "Pergunta de pesquisa"
        )

        st.markdown(
            """
            > **É possível identificar estudantes com maior risco
            > de evasão utilizando informações disponíveis antes
            > da confirmação do abandono?**
            """
        )

        st.write(
            """
            O problema foi tratado como uma classificação binária:

            - **0:** não evasão;
            - **1:** evasão.
            """
        )

    with col2:

        st.subheader(
            "Distribuição da variável-alvo"
        )

        distribuicao = pd.DataFrame(
            {
                "Situação": [
                    "Não evasão",
                    "Evasão"
                ],
                "Percentual": [
                    92.95,
                    7.05
                ]
            }
        ).set_index(
            "Situação"
        )

        st.bar_chart(
            distribuicao
        )

    st.divider()

    st.subheader(
        "Principais decisões metodológicas"
    )

    st.markdown(
        """
        - agregação dos dados por **estudante/ano**;
        - utilização de informações acadêmicas e de engajamento;
        - uso da atividade da plataforma entre **setembro e dezembro**;
        - separação entre treino e teste considerando o estudante;
        - validação cruzada apenas sobre o conjunto de treino;
        - comparação entre Random Forest, XGBoost e LightGBM;
        - tuning com Grid Search e Optuna;
        - conjunto de teste preservado até a avaliação final;
        - ajuste do limiar utilizando apenas treino e validação.
        """
    )

    st.subheader(
        "Fluxo do projeto"
    )

    st.markdown(
        """
        **1. Preparação dos dados**  
        Tratamento, padronização e agregação das bases.

        **2. Análise exploratória**  
        Investigação da evasão, desempenho acadêmico e engajamento.

        **3. Modelagem**  
        Random Forest, XGBoost e LightGBM.

        **4. Otimização**  
        Grid Search e Optuna.

        **5. Avaliação**  
        Validação cruzada, overfitting, conjunto de teste e threshold.

        **6. Aplicação**  
        Disponibilização do modelo em Streamlit.
        """
    )


# ============================================================
# ABA 2 - MODELOS
# ============================================================

with aba_modelos:

    st.header(
        "Comparação dos Modelos"
    )

    st.write(
        """
        Os três algoritmos foram treinados sob o mesmo protocolo
        experimental.

        Cada algoritmo foi avaliado em três configurações:
        baseline, melhor Grid Search e melhor Optuna.
        """
    )

    resultados_modelos = pd.DataFrame(
        {
            "Modelo": [
                "Random Forest",
                "Random Forest",
                "Random Forest",

                "XGBoost",
                "XGBoost",
                "XGBoost",

                "LightGBM",
                "LightGBM",
                "LightGBM"
            ],

            "Configuração": [
                "Baseline",
                "Grid Search",
                "Optuna",

                "Baseline",
                "Grid Search",
                "Optuna",

                "Baseline",
                "Grid Search",
                "Optuna"
            ],

            "F1 Treino": [
                0.9684,
                0.5497,
                0.5550,

                0.4411,
                0.3939,
                0.4822,

                0.2548,
                0.4731,
                0.4956
            ],

            "F1 Validação": [
                0.1537,
                0.3122,
                0.3202,

                0.1898,
                0.3002,
                0.3457,

                0.1656,
                0.3099,
                0.3127
            ]
        }
    )

    st.dataframe(
        resultados_modelos,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "Desempenho na validação cruzada"
    )

    grafico_modelos = resultados_modelos.copy()

    grafico_modelos[
        "Modelo / Configuração"
    ] = (
        grafico_modelos["Modelo"]
        + " - "
        + grafico_modelos["Configuração"]
    )

    grafico_modelos = (
        grafico_modelos
        .set_index(
            "Modelo / Configuração"
        )[
            ["F1 Validação"]
        ]
    )

    st.bar_chart(
        grafico_modelos
    )

    st.success(
        """
        **Modelo selecionado: XGBoost otimizado com Optuna**

        F1 médio na validação cruzada: **0,3457**
        """
    )

    st.divider()

    st.header(
        "Avaliação Final"
    )

    st.write(
        """
        Depois da seleção do XGBoost, o limiar operacional
        foi ajustado utilizando apenas treino e validação.

        O objetivo foi aumentar a capacidade de identificar
        estudantes pertencentes à classe de evasão.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Recall",
            "66,3%"
        )

    with col2:
        st.metric(
            "Precision",
            "20,4%"
        )

    with col3:
        st.metric(
            "F1-Score",
            "0,312"
        )

    with col4:
        st.metric(
            "F2-Score",
            "0,457"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ROC-AUC",
            "0,822"
        )

    with col2:
        st.metric(
            "PR-AUC",
            "0,310"
        )

    with col3:
        st.metric(
            "Limiar operacional",
            "0,34"
        )

    st.divider()

    st.subheader(
        "Matriz de Confusão"
    )

    matriz = pd.DataFrame(
        {
            "Prevê não evasão": [
                9192,
                281
            ],

            "Prevê evasão": [
                2157,
                553
            ]
        },
        index=[
            "Real: não evasão",
            "Real: evasão"
        ]
    )

    st.dataframe(
        matriz,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Evasões reais",
            "834"
        )

        st.metric(
            "Evasões identificadas",
            "553"
        )

    with col2:

        st.metric(
            "Evasões não identificadas",
            "281"
        )

        st.metric(
            "Falsos alertas",
            "2.157"
        )

    st.write(
        """
        O modelo identificou **553 das 834 evasões reais**,
        correspondendo a aproximadamente **66,3%** dos casos.

        Em contrapartida, foram gerados **2.157 falsos alertas**.

        Esse resultado reforça que o modelo deve ser utilizado
        para **priorização de acompanhamento**, e não para
        decisões automáticas sobre estudantes.
        """
    )

    st.warning(
        """
        O F1 final é menor que o F1 utilizado para selecionar
        o modelo porque o limiar foi posteriormente ajustado
        para priorizar Recall e F2.
        """
    )

    st.divider()

    st.subheader(
        "Interpretação do resultado"
    )

    st.write(
        """
        O modelo apresentou capacidade de identificar padrões
        associados à evasão e manteve desempenho semelhante entre
        validação e teste.

        Entretanto, a quantidade de falsos alertas ainda é alta.

        Portanto, a solução deve ser entendida como um protótipo
        de apoio à análise humana.
        """
    )


# ============================================================
# ABA 3 - ANÁLISE DE CASO
# ============================================================

with aba_caso:

    st.header(
        "Análise de um Caso Real"
    )

    st.write(
        """
        Para demonstrar o funcionamento da solução, utilizamos
        uma observação real pertencente ao conjunto de teste.

        Em uma aplicação institucional, os dados seriam obtidos
        diretamente dos sistemas acadêmicos da universidade.
        """
    )

    st.info(
        """
        O modelo utiliza 30 variáveis internamente.

        A interface apresenta apenas alguns indicadores de fácil
        interpretação para facilitar a visualização do caso.
        """
    )

    st.subheader(
        "Resumo acadêmico do estudante"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Ano de ingresso",
            "2014"
        )

        st.metric(
            "Curso na seleção",
            "1ª opção"
        )

        st.metric(
            "Período mais avançado",
            "4º"
        )

    with col2:

        st.metric(
            "Créditos matriculados",
            "55,5"
        )

        st.metric(
            "Desempenho no ano anterior",
            "79,4%"
        )

        st.metric(
            "Créditos de práticas/estágio",
            "9,5"
        )

    with col3:

        st.metric(
            "Eventos na plataforma",
            "180"
        )

        st.metric(
            "Tempo na plataforma",
            "2.926 min"
        )

        st.metric(
            "Interações com recursos",
            "90"
        )

    st.divider()

    if st.button(
        "🔎 Executar análise",
        use_container_width=True
    ):

        score, previsao = prever(
            aluno_base
        )

        st.subheader(
            "Resultado"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Score do modelo",
                f"{score:.3f}"
            )

        with col2:

            st.metric(
                "Limiar de alerta",
                f"{threshold:.2f}"
            )

        with col3:

            st.metric(
                "Classificação",
                (
                    "Alerta"
                    if previsao == 1
                    else "Sem alerta"
                )
            )

        st.progress(
            float(score)
        )

        if previsao == 1:

            st.warning(
                """
                ⚠️ **Alerta de acompanhamento**

                O score produzido pelo modelo ficou acima
                do limiar operacional definido durante
                a validação.

                O estudante seria priorizado para
                acompanhamento acadêmico.
                """
            )

        else:

            st.success(
                """
                ✅ **Sem alerta neste momento**

                O score produzido pelo modelo ficou abaixo
                do limiar operacional.
                """
            )

        st.subheader(
            "Como interpretar?"
        )

        if previsao == 1:

            st.write(
                f"""
                O XGBoost produziu um **score de {score:.3f}**.

                O limiar operacional do sistema é
                **{threshold:.2f}**.

                Como o score está acima desse limiar,
                o sistema gera um alerta para acompanhamento.
                """
            )

        else:

            st.write(
                f"""
                O XGBoost produziu um **score de {score:.3f}**.

                O limiar operacional do sistema é
                **{threshold:.2f}**.

                Como o score está abaixo desse limiar,
                o sistema não gera um alerta.
                """
            )

        st.warning(
            """
            **O score não representa uma probabilidade calibrada
            de abandono.**

            Por exemplo, um score de 0,42 não significa que o
            estudante possui 42% de chance de evadir.

            O valor é utilizado apenas em relação ao limiar
            definido durante a validação do modelo.
            """
        )

        # ----------------------------------------------------
        # TESTE DE PARIDADE
        # ----------------------------------------------------

        score_notebook = 0.419638

        diferenca = abs(
            score - score_notebook
        )

        if diferenca < 0.0001:

            st.success(
                """
                ✅ **Teste de paridade confirmado**

                A previsão apresentada pela aplicação coincide
                com a previsão produzida pelo pipeline no notebook.
                """
            )

            st.caption(
                f"""
                Notebook: {score_notebook:.6f}  
                Streamlit: {score:.6f}
                """
            )

        else:

            st.warning(
                f"""
                Foi encontrada uma diferença de
                {diferenca:.6f}
                em relação ao resultado do notebook.
                """
            )

    st.divider()

    with st.expander(
        "Limitações da solução"
    ):

        st.write(
            """
            O modelo é um protótipo experimental de classificação.

            No conjunto de teste, identificou **66,3% das evasões
            reais**, mas apresentou **Precision de 20,4%**, o que
            significa uma quantidade relevante de falsos alertas.

            Por isso, o resultado não deve ser utilizado para
            decisões automáticas sobre estudantes.

            O uso mais adequado seria como ferramenta de
            triagem, permitindo que equipes acadêmicas priorizem
            estudantes para uma análise humana mais detalhada.
            """
        )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    """
    CP5 — Data Science & Statistical Computing | FIAP 2026
    """
)