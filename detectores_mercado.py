#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 DETECTORES DE MERCADO V8 + ENHANCED - ROYAL SUPREME 👑
💎 DETECTOR MERCADO CAÓTICO + CENÁRIOS EXTREMOS + ANTI-LATERALIZAÇÃO
🔥 FLASH CRASH + PUMP DUMP + ENHANCED OPPORTUNITIES + FILTROS INTELIGENTES
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from collections import deque, defaultdict
from config_royal import ConfigRoyalSupremeEnhanced, MarketScenarioRoyalSupremeEnhanced

class DetectorMercadoCaoticoV8RoyalSupremeEnhanced:
    
    def __init__(self):
        self.historico_analises = deque(maxlen=100)
        
    def detectar_mercado_caotico(self, df: pd.DataFrame, analise_completa: Dict) -> Dict[str, Any]:
        """Detecta se o mercado está em estado caótico"""
        
        if len(df) < 50:
            return {'mercado_caotico': False, 'motivo': 'Dados insuficientes'}
        
        volatilidade = analise_completa.get('volatilidade', 0)
        volume_ratio = analise_completa.get('volume_ratio', 1)
        score_call = analise_completa.get('score_call', 0)
        score_put = analise_completa.get('score_put', 0)
        confluencia_call = analise_completa.get('confluencia_call', 0)
        confluencia_put = analise_completa.get('confluencia_put', 0)
        
        motivos_caotico = []
        
        # 1. Volatilidade extremamente baixa
        if volatilidade < 0.1:
            motivos_caotico.append("Volatilidade extremamente baixa")
        
        # 2. Volume muito abaixo da média
        if volume_ratio < 0.5:
            motivos_caotico.append("Volume muito abaixo da média")
        
        # 3. FILTRO ANTI-LATERALIZAÇÃO ENHANCED
        if len(df) >= ConfigRoyalSupremeEnhanced.LATERALIZACAO_PERIODS:
            ultimas_velas = df.tail(ConfigRoyalSupremeEnhanced.LATERALIZACAO_PERIODS)
            precos_range = ultimas_velas['close'].values
            max_range = np.max(precos_range)
            min_range = np.min(precos_range)
            range_pct = ((max_range - min_range) / min_range) * 100
            
            if range_pct < ConfigRoyalSupremeEnhanced.LATERALIZACAO_THRESHOLD:
                motivos_caotico.append(f"Lateralização extrema - range {range_pct:.3f}%")
        
        # 4. Indicadores muito divergentes
        max_confluencia = max(confluencia_call, confluencia_put)
        if max_confluencia < 15:
            motivos_caotico.append("Indicadores muito divergentes")
        
        # 5. Nenhuma direção clara
        if score_call < 30 and score_put < 30:
            motivos_caotico.append("Nenhuma direção clara")
        
        # 6. Movimento mínimo nas últimas velas
        if len(df) >= 10:
            ultimas_velas = df.tail(10)
            movimentos = []
            for _, vela in ultimas_velas.iterrows():
                movimento = abs(vela['close'] - vela['open']) / vela['open'] * 100
                movimentos.append(movimento)
            
            movimento_medio = np.mean(movimentos)
            if movimento_medio < 0.05:
                motivos_caotico.append("Ranging extremo - velas muito pequenas")
        
        # Determinar se mercado está caótico
        mercado_caotico = len(motivos_caotico) >= 3
        
        # Override por consenso forte
        score_maximo = max(score_call, score_put)
        override_por_consenso = score_maximo >= 80 and max_confluencia >= 20
        
        resultado = {
            'mercado_caotico': mercado_caotico and not override_por_consenso,
            'motivos': motivos_caotico,
            'override_consenso': override_por_consenso,
            'volatilidade': volatilidade,
            'volume_ratio': volume_ratio,
            'max_confluencia': max_confluencia,
            'score_maximo': score_maximo,
            'range_pct': range_pct if len(df) >= ConfigRoyalSupremeEnhanced.LATERALIZACAO_PERIODS else 0
        }
        
        self.historico_analises.append(resultado)
        return resultado

class DetectorCenariosExtremosV8RoyalSupremeEnhanced:
    
    def __init__(self):
        self.historico_cenarios = deque(maxlen=100)
        self.alertas_ativos = {}
        self.protecoes_ativas = {}
        self.contador_anomalias = defaultdict(int)
        
    def detectar_cenario_atual(self, df: pd.DataFrame, par: str) -> MarketScenarioRoyalSupremeEnhanced:
        """Detecta o cenário atual do mercado"""
        if len(df) < 10:
            return MarketScenarioRoyalSupremeEnhanced.NORMAL
        
        movimento_1min = self._calcular_movimento_percentual(df, 1)
        movimento_5min = self._calcular_movimento_percentual(df, 5)
        volume_ratio = self._calcular_volume_ratio(df)
        volatilidade = self._calcular_volatilidade(df)
        
        # ENHANCED OPPORTUNITIES - CONDIÇÕES REALISTAS (CORRIGIDAS)
        
        # TREND OPPORTUNITY
        if (abs(movimento_1min) >= ConfigRoyalSupremeEnhanced.TREND_OPPORTUNITY_MOVEMENT and 
            volume_ratio >= ConfigRoyalSupremeEnhanced.TREND_OPPORTUNITY_VOLUME and 
            volatilidade >= ConfigRoyalSupremeEnhanced.TREND_OPPORTUNITY_VOLATILITY):
            return MarketScenarioRoyalSupremeEnhanced.TREND_OPPORTUNITY
        
        # PULLBACK OPPORTUNITY
        if (abs(movimento_1min) >= ConfigRoyalSupremeEnhanced.PULLBACK_OPPORTUNITY_MOVEMENT and 
            volume_ratio >= ConfigRoyalSupremeEnhanced.PULLBACK_OPPORTUNITY_VOLUME):
            return MarketScenarioRoyalSupremeEnhanced.PULLBACK_OPPORTUNITY
        
        # ELLIOTT OPPORTUNITY
        if (abs(movimento_1min) >= ConfigRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY_MOVEMENT and 
            volume_ratio >= ConfigRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY_VOLUME):
            return MarketScenarioRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY
        
        # ELITE OPPORTUNITY
        if (abs(movimento_1min) >= ConfigRoyalSupremeEnhanced.ELITE_OPPORTUNITY_MOVEMENT and 
            volume_ratio >= ConfigRoyalSupremeEnhanced.ELITE_OPPORTUNITY_VOLUME and 
            volatilidade >= ConfigRoyalSupremeEnhanced.ELITE_OPPORTUNITY_VOLATILITY):
            return MarketScenarioRoyalSupremeEnhanced.ELITE_OPPORTUNITY
        
        # WAVE OPPORTUNITY
        if (abs(movimento_1min) >= ConfigRoyalSupremeEnhanced.WAVE_OPPORTUNITY_MOVEMENT and 
            volume_ratio >= ConfigRoyalSupremeEnhanced.WAVE_OPPORTUNITY_VOLUME):
            return MarketScenarioRoyalSupremeEnhanced.WAVE_OPPORTUNITY
        
        # V8 CENÁRIOS ORIGINAIS (PRESERVADOS)
        
        # FLASH CRASH
        if movimento_1min <= ConfigRoyalSupremeEnhanced.FLASH_CRASH_THRESHOLD:
            return MarketScenarioRoyalSupremeEnhanced.FLASH_CRASH
        
        # PUMP DUMP
        if (movimento_5min >= ConfigRoyalSupremeEnhanced.PUMP_DUMP_THRESHOLD and 
            volume_ratio >= ConfigRoyalSupremeEnhanced.MANIPULATION_VOLUME_THRESHOLD):
            return MarketScenarioRoyalSupremeEnhanced.PUMP_DUMP
        
        # VOLUME ANOMALY
        if volume_ratio >= ConfigRoyalSupremeEnhanced.MANIPULATION_VOLUME_THRESHOLD:
            return MarketScenarioRoyalSupremeEnhanced.VOLUME_ANOMALY
        
        # LIQUIDATION CASCADE
        if movimento_1min <= ConfigRoyalSupremeEnhanced.LIQUIDATION_CASCADE_THRESHOLD:
            return MarketScenarioRoyalSupremeEnhanced.LIQUIDATION_CASCADE
        
        # RANGING DEATH
        if volatilidade <= ConfigRoyalSupremeEnhanced.RANGING_DEATH_THRESHOLD:
            return MarketScenarioRoyalSupremeEnhanced.RANGING_DEATH
        
        # NEWS BOMB
        if self._detectar_gap(df) >= ConfigRoyalSupremeEnhanced.GAP_THRESHOLD:
            return MarketScenarioRoyalSupremeEnhanced.NEWS_BOMB
        
        # MANIPULATION
        if self._detectar_manipulacao(df):
            return MarketScenarioRoyalSupremeEnhanced.MANIPULATION
        
        return MarketScenarioRoyalSupremeEnhanced.NORMAL
    
    def _calcular_movimento_percentual(self, df: pd.DataFrame, periodos: int) -> float:
        """Calcula movimento percentual"""
        if len(df) < periodos:
            return 0.0
        preco_atual = df['close'].iloc[-1]
        preco_anterior = df['close'].iloc[-periodos]
        return ((preco_atual - preco_anterior) / preco_anterior) * 100
    
    def _calcular_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calcula ratio do volume atual vs média"""
        if len(df) < 20:
            return 1.0
        volume_atual = df['volume'].iloc[-1]
        volume_medio = df['volume'].rolling(20).mean().iloc[-1]
        return volume_atual / volume_medio if volume_medio > 0 else 1.0
    
    def _calcular_volatilidade(self, df: pd.DataFrame) -> float:
        """Calcula volatilidade"""
        if len(df) < 20:
            return 0.0
        closes = df['close'].tail(20)
        return (closes.std() / closes.mean()) * 100
    
    def _detectar_gap(self, df: pd.DataFrame) -> float:
        """Detecta gaps de preço"""
        if len(df) < 2:
            return 0.0
        close_anterior = df['close'].iloc[-2]
        open_atual = df['open'].iloc[-1]
        gap = abs(open_atual - close_anterior) / close_anterior * 100
        return gap
    
    def _detectar_manipulacao(self, df: pd.DataFrame) -> bool:
        """Detecta possível manipulação de mercado"""
        if len(df) < 10:
            return False
        volumes_recentes = df['volume'].tail(5)
        volume_medio = df['volume'].tail(20).mean()
        anomalias_consecutivas = sum(1 for v in volumes_recentes if v > volume_medio * 10)
        return anomalias_consecutivas >= 3
    
    def get_cenario_info(self, cenario: MarketScenarioRoyalSupremeEnhanced) -> Dict[str, Any]:
        """Retorna informações sobre o cenário"""
        cenario_info = {
            MarketScenarioRoyalSupremeEnhanced.NORMAL: {
                'descricao': 'Mercado normal',
                'risco': 'BAIXO',
                'acao': 'Operar normalmente'
            },
            MarketScenarioRoyalSupremeEnhanced.TREND_OPPORTUNITY: {
                'descricao': 'Oportunidade de tendência',
                'risco': 'BAIXO',
                'acao': 'Aproveitar movimento'
            },
            MarketScenarioRoyalSupremeEnhanced.PULLBACK_OPPORTUNITY: {
                'descricao': 'Oportunidade de pullback',
                'risco': 'MÉDIO',
                'acao': 'Aguardar confirmação'
            },
            MarketScenarioRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY: {
                'descricao': 'Padrão Elliott detectado',
                'risco': 'MÉDIO',
                'acao': 'Operar com cautela'
            },
            MarketScenarioRoyalSupremeEnhanced.ELITE_OPPORTUNITY: {
                'descricao': 'Oportunidade elite',
                'risco': 'BAIXO',
                'acao': 'Aproveitar oportunidade'
            },
            MarketScenarioRoyalSupremeEnhanced.WAVE_OPPORTUNITY: {
                'descricao': 'Oportunidade de onda',
                'risco': 'MÉDIO',
                'acao': 'Monitorar movimento'
            },
            MarketScenarioRoyalSupremeEnhanced.FLASH_CRASH: {
                'descricao': 'Flash crash detectado',
                'risco': 'MUITO ALTO',
                'acao': 'NÃO OPERAR'
            },
            MarketScenarioRoyalSupremeEnhanced.PUMP_DUMP: {
                'descricao': 'Pump and dump',
                'risco': 'MUITO ALTO',
                'acao': 'NÃO OPERAR'
            },
            MarketScenarioRoyalSupremeEnhanced.MANIPULATION: {
                'descricao': 'Manipulação detectada',
                'risco': 'ALTO',
                'acao': 'EVITAR OPERAÇÕES'
            },
            MarketScenarioRoyalSupremeEnhanced.RANGING_DEATH: {
                'descricao': 'Lateralização extrema',
                'risco': 'ALTO',
                'acao': 'AGUARDAR BREAKOUT'
            },
            MarketScenarioRoyalSupremeEnhanced.VOLUME_ANOMALY: {
                'descricao': 'Anomalia de volume',
                'risco': 'ALTO',
                'acao': 'MONITORAR'
            },
            MarketScenarioRoyalSupremeEnhanced.NEWS_BOMB: {
                'descricao': 'Impacto de notícias',
                'risco': 'ALTO',
                'acao': 'AGUARDAR ESTABILIZAÇÃO'
            },
            MarketScenarioRoyalSupremeEnhanced.LIQUIDATION_CASCADE: {
                'descricao': 'Cascata de liquidação',
                'risco': 'MUITO ALTO',
                'acao': 'NÃO OPERAR'
            }
        }
        
        return cenario_info.get(cenario, {
            'descricao': 'Cenário desconhecido',
            'risco': 'MÉDIO',
            'acao': 'OPERAR COM CAUTELA'
        })
    
    def is_cenario_seguro_para_operar(self, cenario: MarketScenarioRoyalSupremeEnhanced) -> bool:
        """Verifica se o cenário é seguro para operar"""
        cenarios_seguros = [
            MarketScenarioRoyalSupremeEnhanced.NORMAL,
            MarketScenarioRoyalSupremeEnhanced.TREND_OPPORTUNITY,
            MarketScenarioRoyalSupremeEnhanced.PULLBACK_OPPORTUNITY,
            MarketScenarioRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY,
            MarketScenarioRoyalSupremeEnhanced.ELITE_OPPORTUNITY,
            MarketScenarioRoyalSupremeEnhanced.WAVE_OPPORTUNITY
        ]
        
        return cenario in cenarios_seguros
    
    def get_score_boost_por_cenario(self, cenario: MarketScenarioRoyalSupremeEnhanced) -> float:
        """Retorna boost de score baseado no cenário"""
        boost_map = {
            MarketScenarioRoyalSupremeEnhanced.ELITE_OPPORTUNITY: 1.3,
            MarketScenarioRoyalSupremeEnhanced.TREND_OPPORTUNITY: 1.2,
            MarketScenarioRoyalSupremeEnhanced.WAVE_OPPORTUNITY: 1.15,
            MarketScenarioRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY: 1.1,
            MarketScenarioRoyalSupremeEnhanced.PULLBACK_OPPORTUNITY: 1.05,
            MarketScenarioRoyalSupremeEnhanced.NORMAL: 1.0
        }
        
        return boost_map.get(cenario, 0.8)  # Penaliza cenários perigosos
    
    def gerar_relatorio_cenarios(self) -> str:
        """Gera relatório dos cenários detectados"""
        if not self.historico_cenarios:
            return "📊 Nenhum cenário registrado ainda"
        
        contador_cenarios = defaultdict(int)
        for cenario in self.historico_cenarios:
            contador_cenarios[cenario] += 1
        
        relatorio = "📊 RELATÓRIO DE CENÁRIOS DETECTADOS:\n\n"
        
        for cenario, count in sorted(contador_cenarios.items(), key=lambda x: x[1], reverse=True):
            info = self.get_cenario_info(cenario)
            relatorio += f"🔍 {cenario.value}: {count}x - {info['descricao']} (Risco: {info['risco']})\n"
        
        return relatorio

class FiltroAntiLateralizacao:
    """Filtro especializado em detectar e bloquear lateralização"""
    
    @staticmethod
    def detectar_lateralizacao(df: pd.DataFrame, threshold: float = None, periods: int = None) -> Dict[str, Any]:
        """Detecta se o mercado está em lateralização"""
        
        if threshold is None:
            threshold = ConfigRoyalSupremeEnhanced.LATERALIZACAO_THRESHOLD
        if periods is None:
            periods = ConfigRoyalSupremeEnhanced.LATERALIZACAO_PERIODS
        
        if len(df) < periods:
            return {
                'lateralizando': False,
                'range_pct': 0,
                'motivo': 'Dados insuficientes'
            }
        
        # Analisar últimas velas
        ultimas_velas = df.tail(periods)
        precos = ultimas_velas['close'].values
        highs = ultimas_velas['high'].values
        lows = ultimas_velas['low'].values
        
        # Calcular range de preços
        max_preco = np.max(highs)
        min_preco = np.min(lows)
        range_pct = ((max_preco - min_preco) / min_preco) * 100
        
        # Calcular movimento médio das velas
        movimentos_vela = []
        for i in range(len(ultimas_velas)):
            vela = ultimas_velas.iloc[i]
            movimento = abs(vela['close'] - vela['open']) / vela['open'] * 100
            movimentos_vela.append(movimento)
        
        movimento_medio = np.mean(movimentos_vela)
        
        # Determinar se está lateralizando
        lateralizando = range_pct < threshold and movimento_medio < threshold/2
        
        return {
            'lateralizando': lateralizando,
            'range_pct': range_pct,
            'movimento_medio': movimento_medio,
            'threshold_usado': threshold,
            'periods_analisados': periods,
            'motivo': f'Range {range_pct:.3f}% < {threshold}%' if lateralizando else 'Range adequado'
        }
    
    @staticmethod
    def calcular_score_penalizacao(range_pct: float, threshold: float) -> float:
        """Calcula penalização de score baseada na lateralização"""
        if range_pct >= threshold:
            return 1.0  # Sem penalização
        
        # Penalização progressiva
        ratio = range_pct / threshold
        if ratio < 0.3:
            return 0.0  # Bloqueio total
        elif ratio < 0.5:
            return 0.3  # Penalização severa
        elif ratio < 0.7:
            return 0.6  # Penalização moderada
        else:
            return 0.8  # Penalização leve
    
    @staticmethod
    def is_breakout_iminente(df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta se um breakout está iminente"""
        if len(df) < 20:
            return {'breakout_iminente': False, 'direcao': None}
        
        # Analisar compressão de volatilidade
        volatilidades = []
        for i in range(10, len(df)):
            periodo = df.iloc[i-10:i+1]
            vol = periodo['close'].std() / periodo['close'].mean() * 100
            volatilidades.append(vol)
        
        if len(volatilidades) < 5:
            return {'breakout_iminente': False, 'direcao': None}
        
        vol_atual = volatilidades[-1]
        vol_media = np.mean(volatilidades[-5:])
        
        # Volume crescente
        volumes_recentes = df['volume'].tail(5).values
        volume_crescendo = all(volumes_recentes[i] <= volumes_recentes[i+1] for i in range(len(volumes_recentes)-1))
        
        # Compressão de volatilidade + volume crescendo = breakout iminente
        compressao = vol_atual < vol_media * 0.7
        breakout_iminente = compressao and volume_crescendo
        
        # Tentar determinar direção
        direcao = None
        if breakout_iminente:
            preco_atual = df['close'].iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            direcao = 'ALTA' if preco_atual > ma20 else 'BAIXA'
        
        return {
            'breakout_iminente': breakout_iminente,
            'direcao': direcao,
            'compressao_volatilidade': compressao,
            'volume_crescendo': volume_crescendo,
            'vol_atual': vol_atual,
            'vol_media': vol_media
        }

print("✅ DETECTORES DE MERCADO V8 + ENHANCED - ROYAL SUPREME CARREGADO!")
