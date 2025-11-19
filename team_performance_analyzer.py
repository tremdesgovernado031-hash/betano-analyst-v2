import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import poisson
import random

# Configuração da Página Streamlit (A Interface do Aplicativo)
st.set_page_config(layout="wide", page_title="Betano Analyst AI Prototype - Força Ofensiva")

# --- 0. DEFINIÇÃO DE TIMES (Nome Completo e Abreviação para a Simulação) ---

# Dicionário de Times
TIMES = {
    # Brasileirão Série A (20 times)
    'FLA': 'Flamengo', 'PAL': 'Palmeiras', 'SAO': 'São Paulo', 'GRE': 'Grêmio', 
    'CAP': 'Athletico-PR', 'FLU': 'Fluminense', 'COR': 'Corinthians', 'INT': 'Internacional', 
    'BOT': 'Botafogo', 'CAM': 'Atlético-MG', 'BAH': 'Bahia', 'FOR': 'Fortaleza', 
    'CRU': 'Cruzeiro', 'CUI': 'Cuiabá', 'VAS': 'Vasco da Gama', 'VIT': 'Vitória', 
    'JUV': 'Juventude', 'ATL': 'Atlético-GO', 'BGT': 'Red Bull Bragantino', 'ECA': 'Criciúma',

    # La Liga Série A (20 times)
    'RMA': 'Real Madrid', 'BAR': 'Barcelona', 'ATM': 'Atlético de Madrid', 'GIR': 'Girona', 
    'ATH': 'Athletic Bilbao', 'RSO': 'Real Sociedad', 'BET': 'Real Betis', 'VAL': 'Valencia', 
    'VIL': 'Villarreal', 'GET': 'Getafe', 'OSA': 'Osasuna', 'ALA': 'Alavés', 
    'SEV': 'Sevilla', 'CEL': 'Celta de Vigo', 'RAY': 'Rayo Vallecano', 'MLG': 'Mallorca', 
    'CAD': 'Cádiz', 'GRA': 'Granada', 'LPA': 'Las Palmas', 'ALM': 'Almería',

    # Premier League (20 times)
    'MCI': 'Manchester City', 'LIV': 'Liverpool', 'ARS': 'Arsenal', 'TOT': 'Tottenham Hotspur', 
    'CHE': 'Chelsea', 'MUN': 'Manchester United', 'NEW': 'Newcastle United', 'WHU': 'West Ham United', 
    'AVL': 'Aston Villa', 'WOL': 'Wolverhampton', 'CRY': 'Crystal Palace', 'BHA': 'Brighton & Hove Albion', 
    'BRE': 'Brentford', 'EVE': 'Everton', 'FUL': 'Fulham', 'NFO': 'Nottingham Forest', 
    'BOU': 'AFC Bournemouth', 'LEE': 'Leeds United', 'BUR': 'Burnley', 'SHE': 'Sheffield United'
}

# Lista Total de Abreviações (Usada para a simulação e cálculos internos)
TODOS_TIMES_ABR = list(TIMES.keys())

# --- 1. SIMULAÇÃO DE DADOS (Base de Conhecimento da IA) ---

@st.cache_data # Mantém os dados estáveis e evita recalcular em cada interação
def simular_historico_jogos():
    """Cria um DataFrame simulando um histórico de jogos extenso e aleatório para todos os 60 times."""
    
    dados = []
    
    # Geramos um número robusto de jogos (e.g., 600 jogos)
    NUM_JOGOS_SIMULADOS = 600
    
    for _ in range(NUM_JOGOS_SIMULADOS):
        # Escolhe as abreviações
        time_casa_abr = random.choice(TODOS_TIMES_ABR)
        time_fora_abr = random.choice([t for t in TODOS_TIMES_ABR if t != time_casa_abr])
        
        # Simula resultados de gols com um pequeno viés para o time da casa
        gols_casa = random.randint(0, 4) if random.random() < 0.7 else random.randint(0, 2)
        gols_fora = random.randint(0, 2) if random.random() < 0.8 else random.randint(0, 4)
        
        if gols_casa > gols_fora:
            resultado_casa = 'V'
            resultado_fora = 'D'
        elif gols_casa == gols_fora:
            resultado_casa = 'E'
            resultado_fora = 'E'
        else:
            resultado_casa = 'D'
            resultado_fora = 'V'

        # Adiciona o registro do time da casa (usa abreviações internamente)
        dados.append({
            'Time': time_casa_abr, 
            'Adversario': time_fora_abr, 
            'Gols_Feitos': gols_casa, 
            'Gols_Sofridos': gols_fora, 
            'Local': 'C', 
            'Resultado': resultado_casa
        })
        
        # Adiciona o registro do time visitante (usa abreviações internamente)
        dados.append({
            'Time': time_fora_abr, 
            'Adversario': time_casa_abr, 
            'Gols_Feitos': gols_fora, 
            'Gols_Sofridos': gols_casa, 
            'Local': 'F', 
            'Resultado': resultado_fora
        })

    df = pd.DataFrame(dados)
    return df

@st.cache_data
def calcular_forcas(df, time_casa, time_fora):
    """
    Calcula as médias da liga e as forças ofensivas/defensivas (Attack/Defense Strength).
    Recebe as abreviações dos times.
    """
    
    # 1. Cálculo das Médias da Liga (Base)
    media_gols_casa = df[df['Local'] == 'C']['Gols_Feitos'].mean()
    media_gols_fora = df[df['Local'] == 'F']['Gols_Feitos'].mean()
    
    # 2. Cálculo das Médias Específicas dos Times
    df_a = df[df['Time'] == time_casa]
    df_a_casa = df_a[df_a['Local'] == 'C']
    df_b = df[df['Time'] == time_fora]
    df_b_fora = df_b[df_b['Local'] == 'F']
    
    # Garante que as médias específicas existam (evita divisão por zero ou NaN)
    media_feita_a = df_a_casa['Gols_Feitos'].mean() if not df_a_casa.empty else media_gols_casa
    media_sofrida_a = df_a_casa['Gols_Sofridos'].mean() if not df_a_casa.empty else media_gols_fora
    media_feita_b = df_b_fora['Gols_Feitos'].mean() if not df_b_fora.empty else media_gols_fora
    media_sofrida_b = df_b_fora['Gols_Sofridos'].mean() if not df_b_fora.empty else media_gols_casa


    # 3. Cálculo das Forças (STRENGTHS)
    
    # Força Ofensiva do Time A (Casa)
    attack_a = (media_feita_a / media_gols_casa) if media_gols_casa else 1.0
    # Força Defensiva do Time A (Casa) - Quanto o adversário sofre em casa vs. média da liga fora
    defense_a = (media_sofrida_a / media_gols_fora) if media_gols_fora else 1.0
    
    # Força Ofensiva do Time B (Fora)
    attack_b = (media_feita_b / media_gols_fora) if media_gols_fora else 1.0
    # Força Defensiva do Time B (Fora) - Quanto o adversário sofre fora vs. média da liga casa
    defense_b = (media_sofrida_b / media_gols_casa) if media_gols_casa else 1.0
        
    # 4. Cálculo dos Lambdas (Gols Esperados)
    
    # Lambda Time A (Gols esperados para o Time A) = Força_Ataque_A * Força_Defesa_B * Média_Gols_Casa_Liga
    lambda_a = attack_a * defense_b * media_gols_casa

    # Lambda Time B (Gols esperados para o Time B) = Força_Ataque_B * Força_Defesa_A * Média_Gols_Fora_Liga
    lambda_b = attack_b * defense_a * media_gols_fora
    
    # Retorna todos os valores calculados
    return lambda_a, lambda_b, attack_a, defense_a, attack_b, defense_b, media_gols_casa, media_gols_fora

# --- 2. MODELO PREDITIVO (Distribuição de Poisson) ---

@st.cache_data
def calcular_probabilidade_poisson(lambda_a, lambda_b):
    """
    Usa a Distribuição de Poisson para prever a probabilidade de placares.
    """
    max_gols = 5
    prob_matrix = np.zeros((max_gols + 1, max_gols + 1))

    for gols_a in range(max_gols + 1):
        for gols_b in range(max_gols + 1):
            # PMF (Probability Mass Function) de Poisson
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

st.title("⚽ Betano Analyst AI: Protótipo de Análise Preditiva Avançada")
st.subheader("Simulação com 60 Times: Brasileirão, La Liga e Premier League")
st.caption("Este modelo utiliza o método da Força Ofensiva/Defensiva ajustada para calcular as probabilidades de placar exato, 1X2 e Over/Under.")

# 1. Coleta e processamento dos dados
df_historico = simular_historico_jogos()

# 2. Seleção de Times na Interface
st.markdown("####  Seleção da Partida")
col_select_casa, col_select_fora = st.columns(2)

# Usamos os nomes completos nas opções de seleção
TIMES_NOMES = list(TIMES.values())
TIMES_ABREV = list(TIMES.keys())

# Define o índice inicial para Times Casa e Fora
default_casa_index = TIMES_NOMES.index(TIMES['FLA']) # Flamengo
default_fora_index = TIMES_NOMES.index(TIMES['MCI']) # Manchester City

with col_select_casa:
    nome_casa_selecionado = st.selectbox(
        "Time da Casa (Home Team)", 
        options=TIMES_NOMES, 
        index=default_casa_index
    )

with col_select_fora:
    # Filtra os times visitantes para não incluir o time da casa selecionado
    opcoes_fora = [nome for nome in TIMES_NOMES if nome != nome_casa_selecionado]
    
    # Ajusta o índice padrão se o MCI foi o selecionado em casa
    if nome_casa_selecionado == TIMES['MCI']:
        default_fora_index = opcoes_fora.index(TIMES['RMA'])
    else:
        default_fora_index = opcoes_fora.index(TIMES['MCI'])


    nome_fora_selecionado = st.selectbox(
        "Time Visitante (Away Team)", 
        options=opcoes_fora, 
        index=default_fora_index
    )

# 3. Mapeia o nome completo de volta para a abreviação para os cálculos
TIME_CASA_ABR = TIMES_ABREV[TIMES_NOMES.index(nome_casa_selecionado)]
TIME_FORA_ABR = TIMES_ABREV[TIMES_NOMES.index(nome_fora_selecionado)]

# Variáveis de exibição
TIME_CASA_EXIBICAO = nome_casa_selecionado
TIME_FORA_EXIBICAO = nome_fora_selecionado

st.markdown(f"### ⚔️ Confronto Selecionado: {TIME_CASA_EXIBICAO} (Casa) vs {TIME_FORA_EXIBICAO} (Fora)")


# 4. Calcula Forças e Lambdas (usando as abreviações internas)
lambda_a, lambda_b, attack_a, defense_a, attack_b, defense_b, media_liga_c, media_liga_f = calcular_forcas(df_historico, TIME_CASA_ABR, TIME_FORA_ABR)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ⚙️ Forças Ofensivas e Defensivas")
    st.info(f"**{TIME_CASA_EXIBICAO} (Ataque):** {attack_a:.2f} (Base Média Casa: {media_liga_c:.2f} Gols)")
    st.info(f"**{TIME_CASA_EXIBICAO} (Defesa):** {defense_a:.2f} (Base Média Fora: {media_liga_f:.2f} Gols)")
    
with col2:
    st.markdown("#### 🎯 Gols Esperados (Lambdas - Input Poisson)")
    st.success(f"**{TIME_CASA_EXIBICAO} - Lambda (Gols Esperados):** {lambda_a:.2f} Gols")
    st.success(f"**{TIME_FORA_EXIBICAO} - Lambda (Gols Esperados):** {lambda_b:.2f} Gols")

with col3:
    st.markdown("#### ⚙️ Forças Ofensivas e Defensivas")
    st.info(f"**{TIME_FORA_EXIBICAO} (Ataque):** {attack_b:.2f} (Base Média Fora: {media_liga_f:.2f} Gols)")
    st.info(f"**{TIME_FORA_EXIBICAO} (Defesa):** {defense_b:.2f} (Base Média Casa: {media_liga_c:.2f} Gols)")


# 5. Executa o modelo preditivo de Poisson
prob_matrix = calcular_probabilidade_poisson(lambda_a, lambda_b)

st.markdown("---")

st.markdown("#### 🧠 Matriz de Probabilidade de Placar Exato (Modelo Poisson)")
st.caption("A probabilidade de cada placar (em percentual) com base nas forças ajustadas.")

# Formata a matriz para melhor visualização
df_prob_matrix = pd.DataFrame(prob_matrix * 100, 
                              index=[f'{TIME_CASA_EXIBICAO}: {i}' for i in range(prob_matrix.shape[0])], 
                              columns=[f'{TIME_FORA_EXIBICAO}: {j}' for j in range(prob_matrix.shape[1])])

df_prob_matrix = df_prob_matrix.round(2)

st.dataframe(
    df_prob_matrix,
    use_container_width=True,
)


# 6. Cálculo dos Mercados de Apostas (Over/Under e 1X2)
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

# --- Linha de Métricas (Prob. e Odd Justa) ---
col_prob_1, col_prob_X, col_prob_2 = st.columns(3)

with col_prob_1:
    st.metric(label=f"Prob. IA: Vitória {TIME_CASA_EXIBICAO} (1)", value=f"{prob_vitoria_casa * 100:.2f}%")
    st.caption(f"Odd Justa: **{calcular_odd_justa(prob_vitoria_casa):.2f}**")
    
with col_prob_X:
    st.metric(label="Prob. IA: Empate (X)", value=f"{prob_empate * 100:.2f}%")
    st.caption(f"Odd Justa: **{calcular_odd_justa(prob_empate):.2f}**")
    
with col_prob_2:
    st.metric(label=f"Prob. IA: Vitória {TIME_FORA_EXIBICAO} (2)", value=f"{prob_vitoria_fora * 100:.2f}%")
    st.caption(f"Odd Justa: **{calcular_odd_justa(prob_vitoria_fora):.2f}**")

# --- Análise de Value Bet (Input da Odd da Betano) ---
st.markdown("##### Odd da Betano e Value Bet (Insira as Odds Atuais)")

col_odd_1, col_odd_X, col_odd_2 = st.columns(3)

# Análise Vitória Casa (1)
with col_odd_1:
    odd_betano_casa = st.number_input(
        f"Odd Betano (Vitória {TIME_CASA_EXIBICAO})", 
        min_value=1.01, 
        max_value=100.00, 
        value=1.90, # Valor de exemplo
        step=0.01, 
        format="%.2f",
        key='odd_input_1'
    )
    prob_implicita_casa = 1 / odd_betano_casa 
    value_bet_casa = (prob_vitoria_casa - prob_implicita_casa) * 100 
    
    st.markdown("**Análise de Valor**")
    if odd_betano_casa > calcular_odd_justa(prob_vitoria_casa) and value_bet_casa > 1.0:
        st.success(f"VALUE BET IDENTIFICADO! (+{value_bet_casa:.2f}%)")
    else:
        st.warning(f"Odd não compensa o risco. (Valor: {value_bet_casa:.2f}%)")
        
# Análise Empate (X)
with col_odd_X:
    odd_betano_empate = st.number_input(
        "Odd Betano (Empate)", 
        min_value=1.01, 
        max_value=100.00, 
        value=3.50, # Valor de exemplo
        step=0.01, 
        format="%.2f",
        key='odd_input_X'
    )
    prob_implicita_empate = 1 / odd_betano_empate 
    value_bet_empate = (prob_empate - prob_implicita_empate) * 100 
    
    st.markdown("**Análise de Valor**")
    if odd_betano_empate > calcular_odd_justa(prob_empate) and value_bet_empate > 1.0:
        st.success(f"VALUE BET IDENTIFICADO! (+{value_bet_empate:.2f}%)")
    else:
        st.warning(f"Odd não compensa o risco. (Valor: {value_bet_empate:.2f}%)")

# Análise Vitória Fora (2)
with col_odd_2:
    odd_betano_fora = st.number_input(
        f"Odd Betano (Vitória {TIME_FORA_EXIBICAO})", 
        min_value=1.01, 
        max_value=100.00, 
        value=4.00, # Valor de exemplo
        step=0.01, 
        format="%.2f",
        key='odd_input_2'
    )
    prob_implicita_fora = 1 / odd_betano_fora 
    value_bet_fora = (prob_vitoria_fora - prob_implicita_fora) * 100 
    
    st.markdown("**Análise de Valor**")
    if odd_betano_fora > calcular_odd_justa(prob_vitoria_fora) and value_bet_fora > 1.0:
        st.success(f"VALUE BET IDENTIFICADO! (+{value_bet_fora:.2f}%)")
    else:
        st.warning(f"Odd não compensa o risco. (Valor: {value_bet_fora:.2f}%)")


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
        max_value=100.00, 
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
