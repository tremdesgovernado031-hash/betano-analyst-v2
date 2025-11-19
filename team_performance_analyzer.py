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
            'Gols_Feitos':
