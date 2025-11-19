import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import poisson

# Configuração da Página Streamlit (A Interface do Aplicativo)
st.set_page_config(layout="wide", page_title="Betano Analyst AI Prototype - Força Ofensiva")

# --- 1. SIMULAÇÃO DE DADOS (Base de Conhecimento da IA) ---

def simular_historico_jogos():
    """Cria um DataFrame simulando um histórico mais extenso de jogos para cálculo de médias da liga."""
    # Dados Simulados (expandidos para ter uma base de cálculo de média da liga mais rica)
    dados = [
        # Time A (Casa)
        {'Time': 'Time A', 'Adversario': 'X', 'Gols_Feitos': 2, 'Gols_Sofridos': 1, 'Local': 'C', 'Resultado': 'V'},
        {'Time': 'Time A', 'Adversario': 'Y', 'Gols_Feitos': 1, 'Gols_Sofridos': 1, 'Local': 'F', 'Resultado': 'E'},
        {'Time': 'Time A', 'Adversario': 'Z', 'Gols_Feitos': 0, 'Gols_Sofridos': 2, 'Local': 'C', 'Resultado': 'D'},
        {'Time': 'Time A', 'Adversario': 'W', 'Gols_Feitos': 3, 'Gols_Sofridos': 1, 'Local': 'F', 'Resultado': 'V'},
        {'Time': 'Time A', 'Adversario': 'V', 'Gols_Feitos': 1, 'Gols_Sofridos': 0, 'Local': 'C', 'Resultado': 'V'},
        {'Time': 'Time A', 'Adversario': 'T', 'Gols_Feitos': 2, 'Gols_Sofridos': 2, 'Local': 'F', 'Resultado': 'E'},
        {'Time': 'Time A', 'Adversario': 'S', 'Gols_Feitos': 4, 'Gols_Sofridos': 1, 'Local': 'C', 'Resultado': 'V'},
        {'Time': 'Time A', 'Adversario': 'R', 'Gols_Feitos': 0, 'Gols_Sofridos': 1, 'Local': 'F', 'Resultado': 'D'},
        # Time B (Fora)
        {'Time': 'Time B', 'Adversario': 'P', 'Gols_Feitos': 0, 'Gols_Sofridos': 0, 'Local': 'F', 'Resultado': 'E'},
        {'Time': 'Time B', 'Adversario': 'Q', 'Gols_Feitos': 4, 'Gols_Sofridos': 2, 'Local': 'C', 'Resultado': 'V'},
        {'Time': 'Time B', 'Adversario': 'R', 'Gols_Feitos': 1, 'Gols_Sofridos': 3, 'Local': 'F', 'Resultado': 'D'},
        {'Time': 'Time B', 'Adversario': 'S', 'Gols_Feitos': 2, 'Gols_Sofridos': 1, 'Local': 'C', 'Resultado': 'V'},
        {'Time': 'Time B', 'Adversario': 'T', 'Gols_Feitos': 0, 'Gols_Sofridos': 1, 'Local': 'F', 'Resultado': 'D'},
        {'Time': 'Time B', 'Adversario': 'U', 'Gols_Feitos': 3, 'Gols_Sofridos': 0, 'Local': 'C', 'Resultado': 'V'},
        {'Time': 'Time B', 'Adversario': 'V', 'Gols_Feitos': 1, 'Gols_Sofridos': 1, 'Local': 'F', 'Resultado': 'E'},
        {'Time': 'Time B', 'Adversario': 'W', 'Gols_Feitos': 2, 'Gols_Sofridos': 0, 'Local': 'C', 'Resultado': 'V'},
        # Mais jogos aleatórios para média da liga
        {'Time': 'Time C', 'Adversario': 'D', 'Gols_Feitos': 1, 'Gols_Sofridos': 1, 'Local': 'C', 'Resultado': 'E'},
        {'Time': 'Time C', 'Adversario': 'E', 'Gols_Feitos': 2, 'Gols_Sofridos': 3, 'Local': 'F', 'Resultado': 'D'},
        {'Time': 'Time D', 'Adversario': 'C', 'Gols_Feitos': 3, 'Gols_Sofridos': 2, 'Local': 'C', 'Resultado': 'V'},
        {'Time': 'Time D', 'Adversario': 'E', 'Gols_Feitos': 0, 'Gols_Sofridos': 0, 'Local': 'F', 'Resultado': 'E'},
    ]

    df = pd.DataFrame(dados)
    return df

def calcular_forcas(df, time_casa, time_fora):
    """
    Calcula as médias da liga e as forças ofensivas/defensivas (Attack/Defense Strength).
    """
    
    # 1. Cálculo das Médias da Liga (Base)
    
    # Média de Gols Marcados (Feitos) em Casa e Fora
    media_gols_casa = df[df['Local'] == 'C']['Gols_Feitos'].mean()
    media_gols_fora = df[df['Local'] == 'F']['Gols_Feitos'].mean()
    
    # Média global de gols feitos
    # media_total_gols = df['Gols_Feitos'].mean() # Não está sendo usada
    
    # 2. Cálculo das Médias Específicas dos Times
    
    # Dados do Time de Casa (Time A)
    df_a = df[df['Time'] == time_casa]
    df_a_casa = df_a[df_a['Local'] == 'C']
    # df_a_fora = df_a[df_a['Local'] == 'F'] # Não está sendo usada

    # Dados do Time Visitante (Time B)
    df_b = df[df['Time'] == time_fora]
    # df_b_casa = df_b[df_b['Local'] == 'C'] # Não está sendo usada
    df_b_fora = df_b[df_b['Local'] == 'F']

    # 3. Cálculo das Forças (STRENGTHS)
    
    # Força Ofensiva do Time A (Casa)
    attack_a = (df_a_casa['Gols_Feitos'].mean() / media_gols_casa) if media_gols_casa else 1
    # Força Defensiva do Time A (Casa) - Contra gols sofridos pelo visitante
    defense_a = (df_a_casa['Gols_Sofridos'].mean() / media_gols_fora) if media_gols_fora else 1
    
    # Força Ofensiva do Time B (Fora)
    attack_b = (df_b_fora['Gols_Feitos'].mean() / media_gols_fora) if media_gols_fora else 1
    # Força Defensiva do Time B (Fora) - Contra gols sofridos pelo time da casa
    defense_b = (df_b_fora['Gols_Sofridos'].mean() / media_gols_casa) if media_gols_casa else 1
    
    # Garantir que não sejam NaN se a média da liga for zero, embora improvável com dados reais
    if np.isnan(attack_a): attack_a = 1.0
    if np.isnan(defense_a): defense_a = 1.0
    if np.isnan(attack_b): attack_b = 1.0
    if np.isnan(defense_b): defense_b = 1.0
        
    # 4. Cálculo dos Lambdas (Gols Esperados)
    
    # Lambda Time A (Gols esperados para o Time A) = Força_Ataque_A * Força_Defesa_B * Média_Gols_Casa_Liga
    lambda_a = attack_a * defense_b * media_gols_casa

    # Lambda Time B (Gols esperados para o Time B) = Força_Ataque_B * Força_Defesa_A * Média_Gols_Fora_Liga
    lambda_b = attack_b * defense_a * media_gols_fora
    
    return lambda_a, lambda_b, attack_a, defense_a, attack_b, defense_b, media_gols_casa, media_gols_fora

# --- 2. MODELO PREDITIVO (Distribuição de Poisson) ---

def calcular_probabilidade_poisson(lambda_a, lambda_b):
    """
    Usa a Distribuição de Poisson para prever a probabilidade de placares.
    Agora usa lambdas baseados em Forças Ofensivas/Defensivas.
    """
    max_gols = 5
    prob_matrix = np.zeros((max_gols + 1, max_gols + 1))

    for gols_a in range(max_gols + 1):
        for gols_b in range(max_gols + 1):
            # PMF (Probability Mass Function) de Poisson
            # Usa os lambdas calculados por Força * Fraqueza
            prob_a = poisson.pmf(gols_a, lambda_a)
            prob_b = poisson.pmf(gols_b, lambda_b)
            
            # Probabilidade do placar exato (gols_a x gols_b)
            prob_matrix[gols_a, gols_b] = prob_a * prob_b

    return prob_matrix

def calcular_mercados(prob_matrix):
    """Calcula as probabilidades dos mercados Mais/Menos Gols (Over/Under) e 1X2."""
    
    prob_total = prob_matrix.sum() 
    
    prob_vitoria_casa = 0
    prob_empate = 0
    prob_vitoria_fora = 0
    prob_under_2_5_calc = 0
    
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            # 1X2 - Cálculo das probabilidades do resultado final
            if i > j:
                prob_vitoria_casa += prob_matrix[i, j]
            elif i == j:
                prob_empate += prob_matrix[i, j]
            else: # i < j
                prob_vitoria_fora += prob_matrix[i, j]
            
            # Under 2.5 - Cálculo da probabilidade de menos de 3 gols
            if i + j <= 2:
                prob_under_2_5_calc += prob_matrix[i, j]
                
    prob_over_2_5 = prob_total - prob_under_2_5_calc
    
    # Under 1.5 - Cálculo da probabilidade de menos de 2 gols
    prob_under_1_5 = prob_matrix[0, 0] + prob_matrix[0, 1] + prob_matrix[1, 0]
    prob_over_1_5 = prob_total - prob_under_1_5
    
    # Retorna todas as probabilidades (Over/Under e 1X2)
    return prob_over_1_5, prob_over_2_5, prob_vitoria_casa, prob_empate, prob_vitoria_fora

# --- 3. EXECUÇÃO E INTERFACE STREAMLIT ---

# Definição dos Times do Jogo
TIME_CASA = 'Time A'
TIME_FORA = 'Time B'

st.title("⚽ Betano Analyst AI: Protótipo de Análise Preditiva Avançada")
st.subheader(f"Análise: {TIME_CASA} (Casa) vs {TIME_FORA} (Fora) - Modelo Poisson com Força Ofensiva/Defensiva")
st.caption("Este modelo utiliza a Força Ofensiva de cada time contra a Força Defensiva do adversário para calcular os gols esperados (Lambdas), tornando a previsão mais precisa que a média simples.")

# 1. Coleta e processamento dos dados
df_historico = simular_historico_jogos()

# 2. Calcula Forças e Lambdas
lambda_a, lambda_b, attack_a, defense_a, attack_b, defense_b, media_liga_c, media_liga_f = calcular_forcas(df_historico, TIME_CASA, TIME_FORA)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ⚙️ Forças Ofensivas e Defensivas")
    st.info(f"**{TIME_CASA} (Ataque):** {attack_a:.2f} (Base: {media_liga_c:.2f} Gols)")
    st.info(f"**{TIME_CASA} (Defesa):** {defense_a:.2f} (Base: {media_liga_f:.2f} Gols)")
    
with col2:
    st.markdown("#### 🎯 Gols Esperados (Lambdas - Input Poisson)")
    st.success(f"**{TIME_CASA} - Lambda (Gols Esperados):** {lambda_a:.2f} Gols")
    st.success(f"**{TIME_FORA} - Lambda (Gols Esperados):** {lambda_b:.2f} Gols")

with col3:
    st.markdown("#### ⚙️ Forças Ofensivas e Defensivas")
    st.info(f"**{TIME_FORA} (Ataque):** {attack_b:.2f} (Base: {media_liga_f:.2f} Gols)")
    st.info(f"**{TIME_FORA} (Defesa):** {defense_b:.2f} (Base: {media_liga_c:.2f} Gols)")


# 3. Executa o modelo preditivo de Poisson
prob_matrix = calcular_probabilidade_poisson(lambda_a, lambda_b)

st.markdown("---")

st.markdown("#### 🧠 Matriz de Probabilidade de Placar Exato (Modelo Poisson)")
st.caption("A probabilidade de cada placar (em percentual) com base nas forças ajustadas.")

# Formata a matriz para melhor visualização
df_prob_matrix = pd.DataFrame(prob_matrix * 100, 
                              index=[f'{TIME_CASA}: {i}' for i in range(prob_matrix.shape[0])], 
                              columns=[f'{TIME_FORA}: {j}' for j in range(prob_matrix.shape[1])])

# Aplica um gradiente de cor para destacar as maiores probabilidades
st.dataframe(
    df_prob_matrix.style.background_gradient(cmap='plasma', axis=None, subset=pd.IndexSlice[0:5, 0:5]),
    use_container_width=True,
    precision=2
)


# 4. Cálculo dos Mercados de Apostas (Over/Under e 1X2)
prob_over_1_5, prob_over_2_5, prob_vitoria_casa, prob_empate, prob_vitoria_fora = calcular_mercados(prob_matrix)

# Cálculo da ODD JUSTA
def calcular_odd_justa(probabilidade):
    """Calcula a cotação justa (fair odd) como o inverso da probabilidade."""
    if probabilidade > 0:
        return 1 / probabilidade
    return float('inf')


st.markdown("---")
st.markdown("#### 💰 Sugestões para Bilhetes do Dia (Análise de Valor)")

st.markdown("##### ➡️ Análise: Vencedor da Partida (1X2)")

col_1x2_1, col_1x2_2, col_1x2_3, col_1x2_4 = st.columns(4)

# ODD JUSTA para 1 (Vitória Casa)
odd_justa_casa = calcular_odd_justa(prob_vitoria_casa)

with col_1x2_1:
    st.metric(label=f"Prob. da IA: Vitória {TIME_CASA} (1)", value=f"{prob_vitoria_casa * 100:.2f}%")
    
with col_1x2_2:
    st.metric(label="Prob. da IA: Empate (X)", value=f"{prob_empate * 100:.2f}%")
    
with col_1x2_3:
    st.metric(label=f"Prob. da IA: Vitória {TIME_FORA} (2)", value=f"{prob_vitoria_fora * 100:.2f}%")

with col_1x2_4:
    st.metric(label="Odd Justa da IA (Vitória Casa)", value=f"{odd_justa_casa:.2f}")


st.markdown("##### Odd da Betano e Value Bet (Vitória Casa)")
col_odd_1, col_odd_2 = st.columns(2)

with col_odd_1:
    odd_betano_casa = st.number_input(
        f"Odd da Betano (Vitória {TIME_CASA})", 
        min_value=1.01, 
        max_value=10.00, 
        value=1.90, # Valor de exemplo
        step=0.01, 
        format="%.2f",
        key='odd_input_1x2'
    )

with col_odd_2:
    prob_implicita_casa = 1 / odd_betano_casa 
    value_bet_casa = (prob_vitoria_casa - prob_implicita_casa) * 100 
    
    st.markdown("**Análise de Valor (VALUE)**")
    
    if odd_betano_casa > odd_justa_casa and value_bet_casa > 1.0:
        st.success(f"VALUE BET IDENTIFICADO! (+{value_bet_casa:.2f}%)")
        st.markdown(f"**Sugestão:** Apostar na Vitória do {TIME_CASA} (Odd {odd_betano_casa})")
    else:
        st.warning(f"Odd da Betano não compensa o risco calculado pela IA. (Valor: {value_bet_casa:.2f}%)")


st.markdown("---")
st.markdown("##### ➡️ Análise: Mais/Menos Gols (Over/Under)")

odd_justa_over_2_5 = calcular_odd_justa(prob_over_2_5)


col_over_1, col_over_2, col_over_3, col_over_4 = st.columns(4)

with col_over_1:
    st.metric(label="Prob. da IA: Mais de 1.5 Gols", value=f"{prob_over_1_5 * 100:.2f}%")
    
with col_over_2:
    st.metric(label="Prob. da IA: Mais de 2.5 Gols", value=f"{prob_over_2_5 * 100:.2f}%")

with col_over_3:
    st.metric(label="Odd Justa da IA (Over 2.5)", value=f"{odd_justa_over_2_5:.2f}")

with col_over_4:
    # Este é o campo onde você insere a Odd da Betano no dia
    odd_betano_over_2_5 = st.number_input(
        "Odd da Betano (Over 2.5)", 
        min_value=1.01, 
        max_value=10.00, 
        value=2.10, 
        step=0.01, 
        format="%.2f",
        key='odd_input_over'
    ) 
    
    prob_implicita = 1 / odd_betano_over_2_5 
    value_bet = (prob_over_2_5 - prob_implicita) * 100 
    
    st.markdown("**Análise de Valor (VALUE)**")
    
    if odd_betano_over_2_5 > odd_justa_over_2_5 and value_bet > 1.0:
        st.success(f"VALUE BET IDENTIFICADO! (+{value_bet:.2f}%)")
        st.markdown(f"**Sugestão:** Apostar no Over 2.5 Gols (Odd {odd_betano_over_2_5})")
    else:
        st.warning(f"Odd da Betano não compensa o risco calculado pela IA. (Valor: {value_bet:.2f}%)")
