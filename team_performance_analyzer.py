import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import poisson
import random
import time

# Configuração da Página Streamlit (A Interface do Aplicativo)
st.set_page_config(layout="wide", page_title="Betano Analyst AI Prototype - Análise de Valor")

# --- DEFINIÇÕES GLOBAIS ---
# Definimos o Vigorish (Margem de Lucro da Casa de Apostas) para simulação.
VIGORISH_1X2 = 0.05  # 5% de margem no mercado 1X2
VIGORISH_OU = 0.04   # 4% de margem no mercado Over/Under
VIGORISH_BTTS_DC = 0.04 # 4% de margem para BTTS e Dupla Chance
# ---


# --- FUNÇÕES DE ANÁLISE E GESTÃO DE BANCA ---

def calcular_kelly_criterion(prob_modelo, odd_bookie, bankroll):
    """
    Calcula a fração ideal da banca a apostar (Kelly Criterion).
    Limita a fração a 5% para gerenciar o risco e a volatilidade.
    """
    if odd_bookie <= 1.01 or bankroll <= 0:
        return 0.0

    # Probabilidade de Perder (q)
    prob_perder = 1 - prob_modelo
    # Vantagem (b = odd_bookie - 1)
    vantagem = odd_bookie - 1
    
    # Formula Kelly: (b*p - q) / b
    fracao = (vantagem * prob_modelo - prob_perder) / vantagem
    
    if fracao <= 0:
        return 0.0 # Nunca apostar se não houver vantagem (edge)
    
    # Limitamos a fração Kelly a 5% da banca (Limite Pessoal de Risco)
    fracao_limitada = min(fracao, 0.05) 
    
    aposta_sugerida = bankroll * fracao_limitada
    return aposta_sugerida


def calcular_odd_justa(probabilidade):
    """Calcula a cotação justa (fair odd) como o inverso da probabilidade."""
    if probabilidade > 0:
        return 1 / probabilidade
    return float('inf')


def simular_fetch_odds(prob, mercado_tipo, vigorish_percent):
    """
    Simula a busca por odds no mercado (usado para Over/Under, BTTS, etc.).
    """
    fair_odd = calcular_odd_justa(prob)
    # Fator de Vigorish, que simula a margem da casa de apostas
    vigorish_factor = 1 / (1 + vigorish_percent) 
    bookie_odd = fair_odd * vigorish_factor
    
    # Adicionamos um ruído aleatório muito pequeno (max 0.5% de variação)
    noise_factor = 1 + (random.uniform(-0.005, 0.005)) 
    final_odd = bookie_odd * noise_factor
    
    final_odd = max(1.01, final_odd)

    return final_odd # Retornamos a Odd completa (não arredondada)


def calcular_e_ajustar_odds_1x2(prob_vitoria_casa, prob_empate, prob_vitoria_fora, vigorish):
    """
    Calcula as Odds 1X2 ajustando-as pelo True Vigorish.
    """
    probs = [prob_vitoria_casa, prob_empate, prob_vitoria_fora]
    
    # A casa divide o Vigorish proporcionalmente às probabilidades.
    odds_bookie_unadjusted = [1 / (p * (1 + vigorish)) for p in probs]
    
    # Adiciona ruído aleatório individual (muito pequeno)
    odds_final = []
    for odd in odds_bookie_unadjusted:
        noise_factor = 1 + (random.uniform(-0.005, 0.005)) 
        odds_final.append(round(odd * noise_factor, 2))
        
    return odds_final[0], odds_final[1], odds_final[2] # Casa, Empate, Fora


def get_top_n_scores(prob_matrix, n=3):
    """Extrai os N placares mais prováveis da matriz de probabilidade."""
    scores = []
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            scores.append({
                'score': f'{i}-{j}',
                'prob': prob_matrix[i, j]
            })
    
    # Ordena por probabilidade decrescente
    scores.sort(key=lambda x: x['prob'], reverse=True)
    return scores[:n]


# --- 0. DEFINIÇÃO DE TIMES ---

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

# Lista Total de Abreviações
TODOS_TIMES_ABR = list(TIMES.keys())

# --- 1. SIMULAÇÃO DE DADOS ---

@st.cache_data 
def simular_historico_jogos():
    """Cria um DataFrame simulando um histórico de jogos extenso e aleatório."""
    
    dados = []
    NUM_JOGOS_SIMULADOS = 600
    
    for _ in range(NUM_JOGOS_SIMULADOS):
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

        dados.append({'Time': time_casa_abr, 'Adversario': time_fora_abr, 'Gols_Feitos': gols_casa, 'Gols_Sofridos': gols_fora, 'Local': 'C', 'Resultado': resultado_casa})
        dados.append({'Time': time_fora_abr, 'Adversario': time_casa_abr, 'Gols_Feitos': gols_fora, 'Gols_Sofridos': gols_casa, 'Local': 'F', 'Resultado': resultado_fora})

    return pd.DataFrame(dados)

@st.cache_data
def calcular_forcas(df, time_casa, time_fora):
    """Calcula as médias da liga e as forças ofensivas/defensivas (Attack/Defense Strength)."""
    
    media_gols_casa = df[df['Local'] == 'C']['Gols_Feitos'].mean()
    media_gols_fora = df[df['Local'] == 'F']['Gols_Feitos'].mean()
    
    df_a_casa = df[(df['Time'] == time_casa) & (df['Local'] == 'C')]
    df_b_fora = df[(df['Time'] == time_fora) & (df['Local'] == 'F')]
    
    media_feita_a = df_a_casa['Gols_Feitos'].mean() if not df_a_casa.empty else media_gols_casa
    media_sofrida_a = df_a_casa['Gols_Sofridos'].mean() if not df_a_casa.empty else media_gols_fora
    media_feita_b = df_b_fora['Gols_Feitos'].mean() if not df_b_fora.empty else media_gols_fora
    media_sofrida_b = df_b_fora['Gols_Sofridos'].mean() if not df_b_fora.empty else media_gols_casa

    attack_a = (media_feita_a / media_gols_casa) if media_gols_casa else 1.0
    defense_b = (media_sofrida_b / media_gols_casa) if media_gols_casa else 1.0
    
    attack_b = (media_feita_b / media_gols_fora) if media_gols_fora else 1.0
    defense_a = (media_sofrida_a / media_gols_fora) if media_gols_fora else 1.0
        
    lambda_a = attack_a * defense_b * media_gols_casa
    lambda_b = attack_b * defense_a * media_gols_fora
    
    return lambda_a, lambda_b, attack_a, defense_a, attack_b, defense_b, media_gols_casa, media_gols_fora

# --- 2. MODELO PREDITIVO (Distribuição de Poisson) ---

@st.cache_data
def calcular_probabilidade_poisson(lambda_a, lambda_b):
    """Usa a Distribuição de Poisson para prever a probabilidade de placares."""
    # Aumentando o max_gols para calcular Over/Under 4.5
    max_gols = 6 
    prob_matrix = np.zeros((max_gols + 1, max_gols + 1))

    for gols_a in range(max_gols + 1):
        for gols_b in range(max_gols + 1):
            prob_a = poisson.pmf(gols_a, lambda_a)
            prob_b = poisson.pmf(gols_b, lambda_b)
            prob_matrix[gols_a, gols_b] = prob_a * prob_b

    return prob_matrix

def calcular_mercados(prob_matrix):
    """Calcula as probabilidades dos principais mercados usando a matriz Poisson."""
    
    prob_total = prob_matrix.sum() 
    
    # ------------------ 1X2 E DUPLA CHANCE (DC) ------------------
    prob_vitoria_casa = 0
    prob_empate = 0
    prob_vitoria_fora = 0
    
    # ------------------ AMBAS MARCAM (BTTS) ------------------
    prob_btts_sim = 0
    
    # ------------------ OVER/UNDER ------------------
    prob_under_1_5 = 0
    prob_under_2_5 = 0
    prob_under_3_5 = 0
    prob_under_4_5 = 0
    
    # Itera sobre a matriz de placares
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            prob = prob_matrix[i, j]
            soma_gols = i + j
            
            # 1X2 
            if i > j: prob_vitoria_casa += prob
            elif i == j: prob_empate += prob
            else: prob_vitoria_fora += prob
            
            # BTTS SIM (i > 0 E j > 0)
            if i > 0 and j > 0: prob_btts_sim += prob
                
            # UNDER
            if soma_gols <= 1: prob_under_1_5 += prob
            if soma_gols <= 2: prob_under_2_5 += prob
            if soma_gols <= 3: prob_under_3_5 += prob
            if soma_gols <= 4: prob_under_4_5 += prob
                
    # Calcula os OVERs
    prob_over_1_5 = prob_total - prob_under_1_5
    prob_over_2_5 = prob_total - prob_under_2_5
    prob_over_3_5 = prob_total - prob_under_3_5
    prob_over_4_5 = prob_total - prob_under_4_5
    
    # BTTS Não
    prob_btts_nao = prob_total - prob_btts_sim

    # Dupla Chance
    prob_dc_1x = prob_vitoria_casa + prob_empate
    prob_dc_x2 = prob_vitoria_fora + prob_empate
    prob_dc_12 = prob_vitoria_casa + prob_vitoria_fora
    
    # Normaliza as probabilidades 1X2 para somarem 1
    total_1x2 = prob_vitoria_casa + prob_empate + prob_vitoria_fora
    if total_1x2 > 0:
        prob_vitoria_casa /= total_1x2
        prob_empate /= total_1x2
        prob_vitoria_fora /= total_1x2

    return {
        '1': prob_vitoria_casa, 'X': prob_empate, '2': prob_vitoria_fora,
        'DC_1X': prob_dc_1x, 'DC_X2': prob_dc_x2, 'DC_12': prob_dc_12,
        'OU_O1.5': prob_over_1_5, 'OU_U1.5': prob_under_1_5,
        'OU_O2.5': prob_over_2_5, 'OU_U2.5': prob_under_2_5,
        'OU_O3.5': prob_over_3_5, 'OU_U3.5': prob_under_3_5,
        'OU_O4.5': prob_over_4.5, 'OU_U4.5': prob_under_4_5,
        'BTTS_Sim': prob_btts_sim, 'BTTS_Nao': prob_btts_nao,
        # Handicap Asiático (AH 0.0 é a mesma coisa que DNB - Draw No Bet: 1X2 sem empate)
        'AH0.0_1': prob_vitoria_casa / prob_dc_12, # Re-normaliza a probabilidade para 12
        'AH0.0_2': prob_vitoria_fora / prob_dc_12,
    }

# --- 3. EXECUÇÃO E INTERFACE STREAMLIT ---

st.title("⚽ Betano Analyst AI: Protótipo de Análise Preditiva Avançada")
st.subheader("Ferramenta de Uso Pessoal para Encontrar 'Value Bets'")
st.caption("Modelo Poisson com Vigorish ajustado e sugestão de stake via Kelly Criterion.")

# 1. Coleta e processamento dos dados
df_historico = simular_historico_jogos()

# 2. Configuração Pessoal e Seleção de Times
st.sidebar.markdown("### 🏦 Configuração Pessoal")
bankroll_total = st.sidebar.number_input(
    "Valor Total da Sua Banca (R$)", 
    min_value=0.00, 
    value=1000.00, 
    step=100.00, 
    format="%.2f",
    help="Este valor é usado para calcular a Aposta Sugerida via Kelly Criterion."
)
st.sidebar.markdown("---")

st.markdown("####  Seleção da Partida")
col_select_casa, col_select_fora = st.columns(2)

TIMES_NOMES = list(TIMES.values())
TIMES_ABREV = list(TIMES.keys())
default_fla_index = TIMES_NOMES.index(TIMES['FLA'])
default_mci_index = TIMES_NOMES.index(TIMES['MCI'])

with col_select_casa:
    nome_casa_selecionado = st.selectbox("Time da Casa (Home Team)", options=TIMES_NOMES, index=default_fla_index)

with col_select_fora:
    opcoes_fora = [nome for nome in TIMES_NOMES if nome != nome_casa_selecionado]
    try:
        default_fora_index = opcoes_fora.index(TIMES['MCI']) if TIMES['MCI'] in opcoes_fora else 0
    except ValueError:
        default_fora_index = 0

    nome_fora_selecionado = st.selectbox("Time Visitante (Away Team)", options=opcoes_fora, index=default_fora_index)

TIME_CASA_ABR = TIMES_ABREV[TIMES_NOMES.index(nome_casa_selecionado)]
TIME_FORA_ABR = TIMES_ABREV[TIMES_NOMES.index(nome_fora_selecionado)]
TIME_CASA_EXIBICAO = nome_casa_selecionado
TIME_FORA_EXIBICAO = nome_fora_selecionado

st.markdown(f"### ⚔️ Confronto Selecionado: {TIME_CASA_EXIBICAO} (Casa) vs {TIME_FORA_EXIBICAO} (Fora)")


# 4. Calcula Forças e Lambdas 
lambda_a, lambda_b, attack_a, defense_a, attack_b, defense_b, media_liga_c, media_liga_f = calcular_forcas(df_historico, TIME_CASA_ABR, TIME_FORA_ABR)


# 5. Executa o modelo preditivo de Poisson e calcula mercados
prob_matrix = calcular_probabilidade_poisson(lambda_a, lambda_b)
probabilidades = calcular_mercados(prob_matrix)

# Geração das Odds 1X2 ajustadas pelo True Vigorish
odd_betano_casa, odd_betano_empate, odd_betano_fora = calcular_e_ajustar_odds_1x2(
    probabilidades['1'], probabilidades['X'], probabilidades['2'], VIGORISH_1X2
)

# --- Exibição das Métricas Chave (Forças, Lambdas) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ⚙️ Forças Ofensivas e Defensivas")
    st.info(f"**{TIME_CASA_EXIBICAO} (Ataque):** {attack_a:.2f}")
    st.info(f"**{TIME_CASA_EXIBICAO} (Defesa):** {defense_a:.2f}")
    
with col2:
    st.markdown("#### 🎯 Gols Esperados (Lambdas - Input Poisson)")
    st.success(f"**{TIME_CASA_EXIBICAO} - Lambda:** {lambda_a:.2f} Gols")
    st.success(f"**{TIME_FORA_EXIBICAO} - Lambda:** {lambda_b:.2f} Gols")

with col3:
    st.markdown("#### ⚙️ Forças Ofensivas e Defensivas")
    st.info(f"**{TIME_FORA_EXIBICAO} (Ataque):** {attack_b:.2f}")
    st.info(f"**{TIME_FORA_EXIBICAO} (Defesa):** {defense_b:.2f}")

# --- NOVO BLOCO: PROBABILIDADES DE VITÓRIA (1X2) ---
st.markdown("---")
st.markdown("#### 📈 Probabilidades de Resultado (Mercado 1X2)")

col_prob_1, col_prob_X, col_prob_2 = st.columns(3)

with col_prob_1:
    st.metric(label=f"Vitória {TIME_CASA_EXIBICAO} (1)", 
              value=f"{probabilidades['1'] * 100:.2f}%")

with col_prob_X:
    st.metric(label="Empate (X)", 
              value=f"{probabilidades['X'] * 100:.2f}%")

with col_prob_2:
    st.metric(label=f"Vitória {TIME_FORA_EXIBICAO} (2)", 
              value=f"{probabilidades['2'] * 100:.2f}%")
# ----------------------------------------------------

# --- 6. FUNÇÃO DE ANÁLISE DE VALUE BET GERAL ---
def exibir_analise_value(label, prob_modelo, odd_betano, bankroll, is_1x2=True):
    """Função auxiliar para exibir a análise de forma padronizada."""
    
    # Odds Justas e Value
    odd_justa = calcular_odd_justa(prob_modelo)
    prob_implicita = 1 / odd_betano 
    value_bet = (prob_modelo - prob_implicita) * 100 

    # Kelly Criterion
    aposta_sugerida = 0.0
    if odd_betano > odd_justa:
         aposta_sugerida = calcular_kelly_criterion(prob_modelo, odd_betano, bankroll)
    
    st.markdown(f"**{label}:**")
    st.caption(f"Prob. IA: **{prob_modelo * 100:.2f}%** | Odd Justa: **{odd_justa:.2f}** | Odd Simulada: **{odd_betano:.2f}**")

    if aposta_sugerida > 0.0:
        st.success(f"**🔥 VALUE BET ENCONTRADO!** Edge de +{value_bet:.2f}%")
        st.success(f"Aposta Sugerida (Kelly): **R$ {aposta_sugerida:.2f}**")
        st.caption(f"Fração da Banca: {(aposta_sugerida / bankroll_total) * 100:.2f}% (Máximo 5.00%)")
    else:
        # Só exibe se a probabilidade da IA é relevante, para evitar poluição visual de valores baixos
        if prob_modelo * 100 > 5: 
             st.warning(f"Odd não compensa o risco. (Valor: {value_bet:.2f}%)")
        else:
             st.caption("Aposta de baixa probabilidade. (Sem Value Bet)")

st.markdown("---")
st.markdown("#### 💰 Sugestões de Value Bet e Gestão de Banca")

# ====================================================================================
# A. ANÁLISE: VENCEDOR DA PARTIDA (1X2)
# ====================================================================================
st.markdown("##### 1. Mercado: Vencedor da Partida (1X2)")
col_1, col_X, col_2 = st.columns(3)

with col_1:
    exibir_analise_value(
        label=f"Vitória {TIME_CASA_EXIBICAO} (1)",
        prob_modelo=probabilidades['1'],
        odd_betano=odd_betano_casa,
        bankroll=bankroll_total
    )

with col_X:
    exibir_analise_value(
        label="Empate (X)",
        prob_modelo=probabilidades['X'],
        odd_betano=odd_betano_empate,
        bankroll=bankroll_total
    )

with col_2:
    exibir_analise_value(
        label=f"Vitória {TIME_FORA_EXIBICAO} (2)",
        prob_modelo=probabilidades['2'],
        odd_betano=odd_betano_fora,
        bankroll=bankroll_total
    )

# ====================================================================================
# B. ANÁLISE: DUPLA CHANCE (DC)
# ====================================================================================
st.markdown("---")
st.markdown("##### 2. Mercado: Dupla Chance")
col_dc1, col_dc2, col_dc3 = st.columns(3)

# DC 1X
odd_betano_dc1x = round(simular_fetch_odds(probabilidades['DC_1X'], 'DC', VIGORISH_BTTS_DC), 2)
with col_dc1:
    exibir_analise_value(
        label=f"{TIME_CASA_EXIBICAO} ou Empate (1X)",
        prob_modelo=probabilidades['DC_1X'],
        odd_betano=odd_betano_dc1x,
        bankroll=bankroll_total
    )

# DC X2
odd_betano_dcx2 = round(simular_fetch_odds(probabilidades['DC_X2'], 'DC', VIGORISH_BTTS_DC), 2)
with col_dc2:
    exibir_analise_value(
        label=f"Empate ou {TIME_FORA_EXIBICAO} (X2)",
        prob_modelo=probabilidades['DC_X2'],
        odd_betano=odd_betano_dcx2,
        bankroll=bankroll_total
    )

# DC 12
odd_betano_dc12 = round(simular_fetch_odds(probabilidades['DC_12'], 'DC', VIGORISH_BTTS_DC), 2)
with col_dc3:
    exibir_analise_value(
        label="Casa ou Fora (12)",
        prob_modelo=probabilidades['DC_12'],
        odd_betano=odd_betano_dc12,
        bankroll=bankroll_total
    )

# ====================================================================================
# C. ANÁLISE: AMBAS MARCAM (BTTS)
# ====================================================================================
st.markdown("---")
st.markdown("##### 3. Mercado: Ambas as Equipes Marcam (BTTS)")
col_btts_sim, col_btts_nao = st.columns(2)

# BTTS Sim
odd_betano_btts_sim = round(simular_fetch_odds(probabilidades['BTTS_Sim'], 'BTTS', VIGORISH_BTTS_DC), 2)
with col_btts_sim:
    exibir_analise_value(
        label="BTTS - SIM (Ambas Marcam)",
        prob_modelo=probabilidades['BTTS_Sim'],
        odd_betano=odd_betano_btts_sim,
        bankroll=bankroll_total,
        is_1x2=False
    )

# BTTS Não
odd_betano_btts_nao = round(simular_fetch_odds(probabilidades['BTTS_Nao'], 'BTTS', VIGORISH_BTTS_DC), 2)
with col_btts_nao:
    exibir_analise_value(
        label="BTTS - NÃO (Pelo menos 1 time passa em branco)",
        prob_modelo=probabilidades['BTTS_Nao'],
        odd_betano=odd_betano_btts_nao,
        bankroll=bankroll_total,
        is_1x2=False
    )

# ====================================================================================
# D. ANÁLISE: TOTAL DE GOLS (OVER/UNDER)
# ====================================================================================
st.markdown("---")
st.markdown("##### 4. Mercado: Total de Gols (Mais/Menos)")

# --- OVER/UNDER 2.5 (Principal) ---
col_ou_2_5_prob, col_ou_2_5_val = st.columns(2)
odd_betano_over_2_5 = round(simular_fetch_odds(probabilidades['OU_O2.5'], 'OU', VIGORISH_OU), 2)
odd_betano_under_2_5 = round(simular_fetch_odds(probabilidades['OU_U2.5'], 'OU', VIGORISH_OU), 2)

with col_ou_2_5_prob:
    st.metric(label="Prob. Mais de 2.5 Gols", value=f"{probabilidades['OU_O2.5'] * 100:.2f}%")
    st.metric(label="Prob. Menos de 2.5 Gols", value=f"{probabilidades['OU_U2.5'] * 100:.2f}%")

with col_ou_2_5_val:
    exibir_analise_value(
        label="Over 2.5 Gols",
        prob_modelo=probabilidades['OU_O2.5'],
        odd_betano=odd_betano_over_2_5,
        bankroll=bankroll_total,
        is_1x2=False
    )
    exibir_analise_value(
        label="Under 2.5 Gols",
        prob_modelo=probabilidades['OU_U2.5'],
        odd_betano=odd_betano_under_2_5,
        bankroll=bankroll_total,
        is_1x2=False
    )

# --- OVER/UNDER Adicionais (1.5, 3.5, 4.5) em expansor ---
with st.expander("Ver Mais Mercados de Gols (Over/Under 1.5, 3.5, 4.5)"):
    
    st.markdown("###### Over/Under 1.5")
    col_ou_1_5_1, col_ou_1_5_2 = st.columns(2)
    with col_ou_1_5_1:
        odd_betano_over_1_5 = round(simular_fetch_odds(probabilidades['OU_O1.5'], 'OU', VIGORISH_OU), 2)
        exibir_analise_value(label="Over 1.5 Gols", prob_modelo=probabilidades['OU_O1.5'], odd_betano=odd_betano_over_1_5, bankroll=bankroll_total, is_1x2=False)
    with col_ou_1_5_2:
        odd_betano_under_1_5 = round(simular_fetch_odds(probabilidades['OU_U1.5'], 'OU', VIGORISH_OU), 2)
        exibir_analise_value(label="Under 1
