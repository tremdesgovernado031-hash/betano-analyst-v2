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
    Simula a busca por odds no mercado (usado apenas para Over/Under e Placar Exato).
    """
    fair_odd = calcular_odd_justa(prob)
    vigorish_factor = 1 / (1 + vigorish_percent)
    bookie_odd = fair_odd * vigorish_factor
    
    # Adicionamos um ruído aleatório muito pequeno (max 0.5% de variação)
    noise_factor = 1 + (random.uniform(-0.005, 0.005)) 
    final_odd = bookie_odd * noise_factor
    
    final_odd = max(1.01, final_odd)

    return final_odd # Retornamos a Odd completa (não arredondada)


def calcular_e_ajustar_odds_1x2(prob_vitoria_casa, prob_empate, prob_vitoria_fora, vigorish):
    """
    Calcula as Odds 1X2 ajustando-as pelo True Vigorish, garantindo que a soma 
    das probabilidades implícitas seja exatamente (1 + Vigorish).
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
    max_gols = 5
    prob_matrix = np.zeros((max_gols + 1, max_gols + 1))

    for gols_a in range(max_gols + 1):
        for gols_b in range(max_gols + 1):
            prob_a = poisson.pmf(gols_a, lambda_a)
            prob_b = poisson.pmf(gols_b, lambda_b)
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
            # 1X2 
            if i > j: prob_vitoria_casa += prob_matrix[i, j]
            elif i == j: prob_empate += prob_matrix[i, j]
            else: prob_vitoria_fora += prob_matrix[i, j]
            
            # Under 2.5
            if i + j <= 2: prob_under_2_5_calc += prob_matrix[i, j]
                
    prob_over_2_5 = prob_total - prob_under_2_5_calc
    prob_under_1_5 = prob_matrix[0, 0] + prob_matrix[0, 1] + prob_matrix[1, 0]
    prob_over_1_5 = prob_total - prob_under_1_5
    
    # Normaliza as probabilidades 1X2 para somarem 1
    total_1x2 = prob_vitoria_casa + prob_empate + prob_vitoria_fora
    if total_1x2 > 0:
        prob_vitoria_casa /= total_1x2
        prob_empate /= total_1x2
        prob_vitoria_fora /= total_1x2

    return prob_over_1_5, prob_over_2_5, prob_vitoria_casa, prob_empate, prob_vitoria_fora

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
prob_over_1_5, prob_over_2_5, prob_vitoria_casa, prob_empate, prob_vitoria_fora = calcular_mercados(prob_matrix)

# Geração das Odds 1X2 ajustadas pelo True Vigorish
odd_betano_casa, odd_betano_empate, odd_betano_fora = calcular_e_ajustar_odds_1x2(
    prob_vitoria_casa, prob_empate, prob_vitoria_fora, VIGORISH_1X2
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
              value=f"{prob_vitoria_casa * 100:.2f}%")

with col_prob_X:
    st.metric(label="Empate (X)", 
              value=f"{prob_empate * 100:.2f}%")

with col_prob_2:
    st.metric(label=f"Vitória {TIME_FORA_EXIBICAO} (2)", 
              value=f"{prob_vitoria_fora * 100:.2f}%")
# ----------------------------------------------------

# --- 6. ANÁLISE DE VALUE BET ---
st.markdown("---")
st.markdown("#### 💰 Sugestões de Value Bet e Gestão de Banca")

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
        if is_1x2:
             st.info(f"Probabilidade Implícita (Odd) é maior que a calculada pela IA.")
        else:
             st.warning(f"Odd não compensa o risco. (Valor: {value_bet:.2f}%)")


st.markdown("##### ➡️ Análise: Vencedor da Partida (1X2)")
col_prob_1, col_prob_X, col_prob_2 = st.columns(3)

with col_prob_1:
    exibir_analise_value(
        label=f"Vitória {TIME_CASA_EXIBICAO} (1)",
        prob_modelo=prob_vitoria_casa,
        odd_betano=odd_betano_casa,
        bankroll=bankroll_total
    )

with col_prob_X:
    exibir_analise_value(
        label="Empate (X)",
        prob_modelo=prob_empate,
        odd_betano=odd_betano_empate,
        bankroll=bankroll_total
    )

with col_prob_2:
    exibir_analise_value(
        label=f"Vitória {TIME_FORA_EXIBICAO} (2)",
        prob_modelo=prob_vitoria_fora,
        odd_betano=odd_betano_fora,
        bankroll=bankroll_total
    )


st.markdown("---")
st.markdown("##### ➡️ Análise: Mais/Menos Gols (Over/Under)")

# Geração de Odd Over 2.5
odd_justa_over_2_5 = calcular_odd_justa(prob_over_2_5)
odd_betano_over_2_5 = round(simular_fetch_odds(prob_over_2_5, 'Over/Under', VIGORISH_OU), 2)

col_over_1, col_over_2 = st.columns(2)

with col_over_1:
    st.metric(label="Prob. da IA: Mais de 2.5 Gols", value=f"{prob_over_2_5 * 100:.2f}%")

with col_over_2:
    exibir_analise_value(
        label="Over 2.5 Gols",
        prob_modelo=prob_over_2_5,
        odd_betano=odd_betano_over_2_5,
        bankroll=bankroll_total,
        is_1x2=False
    )
    
st.markdown("---")
st.markdown("##### ➡️ Análise: Top 3 Placares Exatos")

top_scores = get_top_n_scores(prob_matrix, n=3)

placar_data = []
for item in top_scores:
    prob = item['prob']
    odd_justa = calcular_odd_justa(prob)
    # Odd para placares exatos costuma ter um vigorish um pouco maior
    odd_simulada = round(simular_fetch_odds(prob, 'Placar', VIGORISH_1X2 + 0.02), 2) 
    
    prob_implicita = 1 / odd_simulada 
    value_bet = (prob - prob_implicita) * 100 
    
    analise_valor = f"{value_bet:.2f}%"
    aposta = 0.0
    
    if odd_simulada > odd_justa:
        aposta = calcular_kelly_criterion(prob, odd_simulada, bankroll_total)
        analise_valor = f"🔥 VALUE BET ({analise_valor})!"

    placar_data.append({
        "Placar": item['score'],
        "Prob. IA (%)": round(prob * 100, 2),
        "Odd Justa": round(odd_justa, 2),
        "Odd Simulada": odd_simulada,
        "Análise de Valor": analise_valor,
        "Aposta Sugerida (R$)": round(aposta, 2) if aposta > 0.0 else "R$ 0.00"
    })

df_placar = pd.DataFrame(placar_data)
st.dataframe(df_placar, use_container_width=True, hide_index=True)


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
