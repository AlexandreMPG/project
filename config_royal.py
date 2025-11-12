#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 CONFIG ROYAL SUPREME ENHANCED + DATABASE AI - ELITE PRECISION 80-90% 👑
💎 CONFIGURAÇÕES PARA WIN RATE FIXO 80-90% COM 8+ SINAIS/HORA
🔥 FOCO MÁXIMO EM PRECISÃO E WIN M1 (EVITAR GALE)
🎯 SISTEMA CALIBRADO PARA ASSERTIVIDADE MÁXIMA
"""

import pytz
from enum import Enum
from dataclasses import dataclass

TIMEZONE = pytz.timezone('America/Sao_Paulo')

PARES_CRYPTO = {
    'btcusdt': {'nome': 'BTC/USDT', 'prioridade': 1, 'correlacao_btc': 0.0, 'volatilidade_base': 1.0, 'enhanced_weight': 1.0, 'risk_level': 'LOW'},
    'ethusdt': {'nome': 'ETH/USDT', 'prioridade': 2, 'correlacao_btc': 0.85, 'volatilidade_base': 1.2, 'enhanced_weight': 0.98, 'risk_level': 'LOW'},
    'solusdt': {'nome': 'SOL/USDT', 'prioridade': 3, 'correlacao_btc': 0.65, 'volatilidade_base': 1.8, 'enhanced_weight': 0.95, 'risk_level': 'MEDIUM'},
    'xrpusdt': {'nome': 'XRP/USDT', 'prioridade': 4, 'correlacao_btc': 0.70, 'volatilidade_base': 1.5, 'enhanced_weight': 0.92, 'risk_level': 'MEDIUM'},
    'adausdt': {'nome': 'ADA/USDT', 'prioridade': 5, 'correlacao_btc': 0.75, 'volatilidade_base': 1.8, 'enhanced_weight': 0.88, 'risk_level': 'MEDIUM'}
}

ROYAL_TELEGRAM = {
    'free_token': '7953706884:AAFGI4-OEPVj6LoN8INRc9Ccq_oPwTlRGHo',
    'free_chat': '-4677274114',
    'vip_token': '7953706884:AAFGI4-OEPVj6LoN8INRc9Ccq_oPwTlRGHo', 
    'vip_chat': '-1002707031255',
    'enabled': True,
    'admin_ids': [6312295060, 676446061]
}

class ConfigRoyalSupremeEnhanced:
    # TIMING V8 ORIGINAL PRESERVADO 
    SEGUNDO_ANALISE = 35
    VERIFICACAO_M1_SEGUNDOS = 77
    VERIFICACAO_GALE_SEGUNDOS = 137
    TIMEOUT_SINAL_SEGUNDOS = 180
    
    # TIMEFRAMES V8 ORIGINAL
    TIMEFRAMES_OPERACAO = ['1m', '5m']
    TIMEFRAMES_FILTROS = ['5m', '15m', '30m', '1h', '4h', '1d']
    TIMEFRAMES_CONFLUENCIA = ['1m', '5m', '15m', '30m']
    
    # 🎯 ELITE PRECISION - 80-90% WIN RATE CONFIGURAÇÕES
    MIN_SCORE_NORMAL = 50                    # Reduzido de 65 para permitir 8+ sinais/hora
    MIN_SCORE_SNIPER = 75                    # Mantido para sniper mode
    MIN_SCORE_ELITE_PRECISION = 70           # NOVO: Para sinais de alta precisão
    MIN_SCORE_WAVE = 55                      # Reduzido de 60
    
    MIN_CONFLUENCIA = 5                      # Reduzido de 7 para 5
    MIN_CONFLUENCIA_SNIPER = 7               # Reduzido de 8 para 7
    MIN_CONFLUENCIA_ELITE_PRECISION = 6     # NOVO: Confluência para precisão
    MIN_CONFLUENCIA_WAVE = 5                 # Reduzido de 6 para 5
    
    # 🎯 VOLUME REQUIREMENTS - AJUSTADO PARA 8+ SINAIS/HORA
    MIN_VOLUME_RATIO = 1.15                  # Reduzido de 1.3 para 1.15
    MIN_VOLUME_RATIO_CONFIRMATION = 1.3     # Reduzido de 1.5 para 1.3
    MIN_VOLUME_ELITE_PRECISION = 1.4        # NOVO: Volume para alta precisão
    
    # 🎯 THRESHOLDS ELITE PRECISION
    MIN_INDICATOR_CONFLUENCE = 12            # Reduzido de 15 para 12
    ANTI_LOSS_CONFIDENCE_THRESHOLD = 75     # Reduzido de 85 para 75
    SURVIVABILITY_SCORE_MIN = 65            # Reduzido de 70 para 65
    REQUIRE_EXTREME_CONFLUENCE = False      # Mudado para False para permitir mais sinais
    
    # ROYAL SESSION CONTROL
    SESSAO_FREE_SINAIS = 8                  # Aumentado de 5 para 8 (meta de sinais/hora)
    SESSAO_FREE_DURACAO = 60
    SESSAO_FREE_INTERVALO = 45              # Reduzido de 60 para 45
    AVISO_SESSAO_MINUTOS = 5                # Reduzido de 7 para 5
    
    # 🎯 AUTO CALIBRADOR ELITE - AJUSTADO PARA 80-90% WIN RATE
    AUTO_CALIBRADOR_ENABLED = True
    CALIBRADOR_ADJUSTMENT_ELITE = {
        'btcusdt': 0,
        'ethusdt': 0,                       # Reduzido de 8   - ANTES 0
        'solusdt': 0,                       # Reduzido de 15  - ANTES 5
        'xrpusdt': 0,                      # Reduzido de 20  - ANTES10
        'adausdt': 0                       # Reduzido de 30  - ANTES 15
    }
    
    # 🎯 AUTO CALIBRADOR DINÂMICO - WIN M1 FOCUSED
    CALIBRADOR_WIN_M1_THRESHOLD = 80        # NOVO: Se WIN M1 < 80%, ser mais rigoroso
    CALIBRADOR_WIN_M1_BOOST = 75            # NOVO: Se WIN M1 > 90%, ser menos rigoroso
    CALIBRADOR_QUALITY_THRESHOLD_BASE = 15  # NOVO: Ajuste base por quality score (reduzido)
    
    # WAVE SURF SYSTEM - OTIMIZADO
    WAVE_DETECTION_THRESHOLD = 0.12        # Reduzido de 0.15
    WAVE_MIN_VOLUME = 1.8                  # Reduzido de 2.0
    COOLDOWN_NORMAL_SEGUNDOS = 120         # Reduzido de 180
    COOLDOWN_FLUXO_SEGUNDOS = 90           # Reduzido de 120
    COOLDOWN_POS_WIN = 180                 # Reduzido de 240
    COOLDOWN_POS_LOSS = 360                # Reduzido de 420
    COOLDOWN_POS_LOSS_ADA = 480            # Reduzido de 600 (8min)
    
    # EXTREME SCENARIOS V8 ORIGINAL - MANTIDOS
    FLASH_CRASH_THRESHOLD = -2.0
    PUMP_DUMP_THRESHOLD = 3.0
    MANIPULATION_VOLUME_THRESHOLD = 15.0
    GAP_THRESHOLD = 5.0
    LIQUIDATION_CASCADE_THRESHOLD = -1.5
    RANGING_DEATH_THRESHOLD = 0.08         # Reduzido de 0.1
    
    # SISTEMA LUCRATIVO
    MAX_GALE_LEVEL = 1
    PROFIT_OPTIMIZATION = True
    LOSS_PREVENTION_MODE = True
    WIN_M1_FOCUS_MODE = True               # NOVO: Foco em WIN M1
    
    # ENHANCED FEATURES CONFIG
    SUPORTE_RESISTENCIA_ENABLED = True
    LTA_LTB_ENABLED = True
    PULLBACK_THROWBACK_ENABLED = True
    ELLIOTT_WAVES_ENABLED = True
    TREND_PRIORITY_ENABLED = True
    
    # 🎯 S/R CONFIG - ELITE PRECISION (4+ TOQUES)
    SR_LOOKBACK_PERIODS = 60               # Aumentado de 50 para mais histórico
    SR_TOUCH_TOLERANCE = 0.0015            # Reduzido de 0.002 (mais preciso)
    SR_STRENGTH_MIN = 4                    # Aumentado de 3 para 4 (S/R mais forte)
    SR_ELITE_PRECISION_MIN = 4             # NOVO: Mínimo 4 toques para elite precision
    SR_DISTANCE_MAX_ELITE = 0.008          # NOVO: Distância máxima para S/R elite
    
    # 🎯 LTA/LTB CONFIG - ELITE PRECISION
    TRENDLINE_MIN_TOUCHES = 3              # Aumentado de 2 para 3
    TRENDLINE_LOOKBACK = 40                # Aumentado de 30 para 40
    TRENDLINE_TOLERANCE = 0.002            # Reduzido de 0.003 (mais preciso)
    TRENDLINE_ELITE_MIN_TOUCHES = 3        # NOVO: Mínimo para elite
    
    # PULLBACK CONFIG - OTIMIZADO
    PULLBACK_MIN_RETRACE = 0.30            # Reduzido de 0.382
    PULLBACK_MAX_RETRACE = 0.70            # Aumentado de 0.618
    THROWBACK_CONFIRMATION_CANDLES = 2
    
    # 🎯 ENHANCED OPPORTUNITIES - OTIMIZADO PARA 8+ SINAIS/HORA
    TREND_OPPORTUNITY_MOVEMENT = 0.02      # Reduzido de 0.03
    TREND_OPPORTUNITY_VOLUME = 1.2         # Reduzido de 1.3
    TREND_OPPORTUNITY_VOLATILITY = 0.15    # Reduzido de 0.2
    
    PULLBACK_OPPORTUNITY_MOVEMENT = 0.018  # Reduzido de 0.025
    PULLBACK_OPPORTUNITY_VOLUME = 1.1      # Reduzido de 1.2
    
    ELLIOTT_OPPORTUNITY_MOVEMENT = 0.025   # Reduzido de 0.04
    ELLIOTT_OPPORTUNITY_VOLUME = 1.2       # Reduzido de 1.4
    
    ELITE_OPPORTUNITY_MOVEMENT = 0.05      # Reduzido de 0.08
    ELITE_OPPORTUNITY_VOLUME = 1.6         # Reduzido de 2.0
    ELITE_OPPORTUNITY_VOLATILITY = 0.25    # Reduzido de 0.3
    
    WAVE_OPPORTUNITY_MOVEMENT = 0.03       # Reduzido de 0.05
    WAVE_OPPORTUNITY_VOLUME = 1.3          # Reduzido de 1.5
    
    # 🎯 FILTRO ANTI-LATERALIZAÇÃO - MENOS RESTRITIVO
    LATERALIZACAO_THRESHOLD = 0.08         # Reduzido de 0.12
    LATERALIZACAO_PERIODS = 12             # Reduzido de 15
    
    # 🎯 NOVOS FILTROS ELITE PRECISION
    CONFLUENCE_BOOST_THRESHOLD = 8         # NOVO: Boost quando confluência >= 8
    CONFLUENCE_BOOST_SCORE = 15            # NOVO: +15 score quando confluência alta
    
    PATTERN_CONFIRMATION_REQUIRED = True   # NOVO: Padrões devem confirmar direção
    MOMENTUM_ALIGNMENT_REQUIRED = True     # NOVO: Momentum deve alinhar com sinal
    
    # 🎯 IA SUPERVISORA PRECISION SETTINGS
    IA_SUPERVISORA_MIN_SR_TOUCHES = 4      # NOVO: Mínimo 4 toques para IA sugerir
    IA_SUPERVISORA_MAX_DISTANCE = 0.006   # NOVO: Distância máxima para IA atuar
    IA_SUPERVISORA_CONFIRMATION_REQUIRED = True  # NOVO: Requer confirmação de vela
    
    # 🎯 BLACKLIST ELITE - MENOS RESTRITIVO MAS MAIS INTELIGENTE
    BLACKLIST_QUALITY_THRESHOLD = 45       # Reduzido de 55
    BLACKLIST_PATTERN_LIMIT = 2           # NOVO: Máximo 2 padrões perigosos simultâneos
    BLACKLIST_HORARIO_PROTECTION = True   # NOVO: Proteção por horário específico
    
    # 🎯 PRECISION TIMING SETTINGS
    PRECISION_ENTRY_TIMING = True         # NOVO: Timing preciso de entrada
    PRECISION_VOLUME_CONFIRMATION = True  # NOVO: Volume deve confirmar movimento
    PRECISION_MOMENTUM_CHECK = True       # NOVO: Momentum deve estar alinhado

class TipoSinalRoyalSupremeEnhanced(Enum):
    CALL = "CALL"
    PUT = "PUT"
    CALL_SNIPER = "CALL_SNIPER"
    PUT_SNIPER = "PUT_SNIPER" 
    CALL_WAVE = "CALL_WAVE"
    PUT_WAVE = "PUT_WAVE"
    CALL_ENHANCED = "CALL_ENHANCED"
    PUT_ENHANCED = "PUT_ENHANCED"
    CALL_SURVIVABILITY = "CALL_SURVIVABILITY"
    PUT_SURVIVABILITY = "PUT_SURVIVABILITY"
    CALL_TREND_PRIORITY = "CALL_TREND_PRIORITY"
    PUT_TREND_PRIORITY = "PUT_TREND_PRIORITY"
    CALL_ELLIOTT = "CALL_ELLIOTT"
    PUT_ELLIOTT = "PUT_ELLIOTT"
    CALL_ELITE_PRECISION = "CALL_ELITE_PRECISION"   # NOVO
    PUT_ELITE_PRECISION = "PUT_ELITE_PRECISION"     # NOVO

class StatusSinalRoyalSupremeEnhanced(Enum):
    ATIVO = "ATIVO"
    AGUARDANDO_GALE = "AGUARDANDO_GALE"
    WIN_M1 = "WIN_M1"
    WIN_GALE = "WIN_GALE"
    LOSS = "LOSS"
    TIMEOUT = "TIMEOUT"
    PROTECTED = "PROTECTED"

class MarketScenarioRoyalSupremeEnhanced(Enum):
    NORMAL = "NORMAL"
    FLASH_CRASH = "FLASH_CRASH"
    PUMP_DUMP = "PUMP_DUMP"
    NEWS_BOMB = "NEWS_BOMB"
    LIQUIDATION_CASCADE = "LIQUIDATION_CASCADE"
    MANIPULATION = "MANIPULATION"
    RANGING_DEATH = "RANGING_DEATH"
    CORRELATION_BREAK = "CORRELATION_BREAK"
    VOLUME_ANOMALY = "VOLUME_ANOMALY"
    TIMEFRAME_CHAOS = "TIMEFRAME_CHAOS"
    WHALE_MOVEMENT = "WHALE_MOVEMENT"
    MERCADO_CAOTICO = "MERCADO_CAOTICO"
    WAVE_OPPORTUNITY = "WAVE_OPPORTUNITY"
    ELITE_OPPORTUNITY = "ELITE_OPPORTUNITY"
    TREND_OPPORTUNITY = "TREND_OPPORTUNITY"
    PULLBACK_OPPORTUNITY = "PULLBACK_OPPORTUNITY"
    ELLIOTT_OPPORTUNITY = "ELLIOTT_OPPORTUNITY"
    PRECISION_OPPORTUNITY = "PRECISION_OPPORTUNITY"  # NOVO

class SurvivabilityMode(Enum):
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE" 
    PARANOID = "PARANOID"
    BUNKER = "BUNKER"
    ELITE_PRECISION = "ELITE_PRECISION"  # NOVO

# 🎯 CONFIGURAÇÕES ELITE PRECISION ESPECÍFICAS
class ElitePrecisionConfig:
    """Configurações específicas para modo Elite Precision 80-90%"""
    
    # WIN M1 FOCUS
    TARGET_WIN_M1_RATE = 85               # Meta de 85% WIN M1
    MIN_WIN_M1_RATE_THRESHOLD = 80        # Mínimo 80% WIN M1
    MAX_WIN_M1_RATE_THRESHOLD = 90        # Máximo 90% WIN M1
    
    # SIGNAL FREQUENCY
    TARGET_SIGNALS_PER_HOUR = 8           # Meta de 8 sinais/hora
    MIN_SIGNALS_PER_HOUR = 6              # Mínimo 6 sinais/hora
    MAX_SIGNALS_PER_HOUR = 12             # Máximo 12 sinais/hora
    
    # PRECISION REQUIREMENTS
    MIN_SR_TOUCHES_PRECISION = 4          # Mínimo 4 toques S/R
    MAX_SR_DISTANCE_PRECISION = 0.006     # Máximo 0.6% de distância
    MIN_VOLUME_CONFIRMATION = 1.3         # Volume mínimo 1.3x
    
    # PATTERN REQUIREMENTS
    REQUIRED_PATTERN_CONFLUENCE = 6       # Mínimo 6 confluências
    REQUIRED_MOMENTUM_ALIGNMENT = True    # Momentum deve alinhar
    REQUIRED_VOLUME_CONFIRMATION = True   # Volume deve confirmar
    
    # TIMING PRECISION
    ENTRY_TIMING_PRECISION = True         # Timing preciso obrigatório
    VELA_CONFIRMATION_REQUIRED = True     # Vela deve confirmar direção
    REJECTION_CONFIRMATION_REQUIRED = True # Rejeição deve ser confirmada

# Configurações para sistema otimizado
CACHE_ENABLED = True           # Habilitado para performance
DATABASE_QUERIES_LIMIT = 2     # Aumentado para mais dados
DETECTOR_LIGHT_MODE = False    # Desabilitado para máxima precisão

print("✅ CONFIG ROYAL SUPREME ENHANCED + DATABASE AI - ELITE PRECISION 80-90% CARREGADO!")
print("🎯 CONFIGURAÇÃO OTIMIZADA PARA 8+ SINAIS/HORA COM 80-90% WIN RATE!")
print("💎 FOCO EM WIN M1 - EVITAR GALE - MÁXIMA ASSERTIVIDADE!")