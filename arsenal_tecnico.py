#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 ARSENAL TÉCNICO COMPLETO V8 - ROYAL SUPREME ENHANCED 👑
💎 15 INDICADORES TÉCNICOS CALIBRADOS
🔥 RSI + MACD + STOCH + WILLIAMS + VWAP + BB + ADX + CCI + MOMENTUM + ROC + MFI + TSI + UO + PSAR + AO
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from collections import defaultdict

class ArsenalTecnicoCompletoV8RoyalSupremeEnhanced:
    
    def __init__(self):
        self.cache_indicadores = defaultdict(dict)
    
    @staticmethod
    def ema(dados: np.array, periodo: int) -> float:
        """Exponential Moving Average"""
        if len(dados) < periodo:
            return np.mean(dados) if len(dados) > 0 else 0
        multiplicador = 2 / (periodo + 1)
        ema = dados[0]
        for preco in dados[1:]:
            ema = (preco * multiplicador) + (ema * (1 - multiplicador))
        return ema
    
    @staticmethod
    def sma(dados: np.array, periodo: int) -> float:
        """Simple Moving Average"""
        if len(dados) < periodo:
            return np.mean(dados) if len(dados) > 0 else 0
        return np.mean(dados[-periodo:])
    
    @staticmethod
    def rsi(dados: np.array, periodo: int = 14) -> float:
        """Relative Strength Index"""
        if len(dados) < periodo + 1:
            return 50
        deltas = np.diff(dados)
        ganhos = np.where(deltas > 0, deltas, 0)
        perdas = np.where(deltas < 0, -deltas, 0)
        ganho_medio = np.mean(ganhos[-periodo:])
        perda_media = np.mean(perdas[-periodo:])
        if perda_media == 0:
            return 100
        rs = ganho_medio / perda_media
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(dados: np.array, rapido: int = 12, lento: int = 26, sinal: int = 9) -> Dict:
        """MACD - Moving Average Convergence Divergence"""
        if len(dados) < lento:
            return {'linha': 0, 'sinal': 0, 'histograma': 0, 'momentum': 0}
        
        ema_rapida = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced.ema(dados, rapido)
        ema_lenta = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced.ema(dados, lento)
        linha_macd = ema_rapida - ema_lenta
        
        if len(dados) >= lento + sinal:
            macd_values = []
            for i in range(sinal):
                idx = len(dados) - sinal + i
                if idx >= lento:
                    ema_r = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced.ema(dados[:idx+1], rapido)
                    ema_l = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced.ema(dados[:idx+1], lento)
                    macd_values.append(ema_r - ema_l)
            linha_sinal = np.mean(macd_values) if macd_values else linha_macd
        else:
            linha_sinal = linha_macd
        
        histograma = linha_macd - linha_sinal
        momentum = 1 if histograma > 0 else -1
        
        return {'linha': linha_macd, 'sinal': linha_sinal, 'histograma': histograma, 'momentum': momentum}
    
    @staticmethod
    def stochastic(df: pd.DataFrame, periodo: int = 14) -> Dict:
        """Stochastic Oscillator"""
        if len(df) < periodo:
            return {'k': 50, 'd': 50, 'overbought': False, 'oversold': False}
        
        highs = df['high'].values[-periodo:]
        lows = df['low'].values[-periodo:]
        close = df['close'].iloc[-1]
        
        highest_high = np.max(highs)
        lowest_low = np.min(lows)
        
        if highest_high == lowest_low:
            k = 50
        else:
            k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        d = k  # Simplificado
        
        return {
            'k': k,
            'd': d,
            'overbought': k > 80,
            'oversold': k < 20
        }
    
    @staticmethod
    def williams_r(df: pd.DataFrame, periodo: int = 14) -> float:
        """Williams %R"""
        if len(df) < periodo:
            return -50
        
        highs = df['high'].values[-periodo:]
        lows = df['low'].values[-periodo:]
        close = df['close'].iloc[-1]
        
        highest_high = np.max(highs)
        lowest_low = np.min(lows)
        
        if highest_high == lowest_low:
            return -50
        
        wr = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return wr
    
    @staticmethod
    def vwap(df: pd.DataFrame) -> float:
        """Volume Weighted Average Price"""
        if len(df) < 2:
            return df['close'].iloc[-1] if len(df) > 0 else 0
        tp = (df['high'] + df['low'] + df['close']) / 3
        pv = tp * df['volume']
        return pv.sum() / df['volume'].sum() if df['volume'].sum() > 0 else tp.mean()
    
    @staticmethod
    def bandas_bollinger(dados: np.array, periodo: int = 20, desvio: float = 2) -> Dict:
        """Bollinger Bands"""
        if len(dados) < periodo:
            media = np.mean(dados) if len(dados) > 0 else 0
            return {'superior': media, 'media': media, 'inferior': media, 'posicao': 0.5}
        media = np.mean(dados[-periodo:])
        std = np.std(dados[-periodo:])
        superior = media + (std * desvio)
        inferior = media - (std * desvio)
        posicao = (dados[-1] - inferior) / (superior - inferior) if superior != inferior else 0.5
        return {'superior': superior, 'media': media, 'inferior': inferior, 'posicao': posicao}
    
    @staticmethod
    def adx(df: pd.DataFrame, periodo: int = 14) -> float:
        """Average Directional Index"""
        if len(df) < periodo + 1:
            return 0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        if len(tr_list) < periodo:
            return 0
        
        dm_plus = []
        dm_minus = []
        
        for i in range(1, len(high)):
            move_up = high[i] - high[i-1]
            move_down = low[i-1] - low[i]
            
            dm_plus.append(move_up if move_up > move_down and move_up > 0 else 0)
            dm_minus.append(move_down if move_down > move_up and move_down > 0 else 0)
        
        if len(dm_plus) < periodo or len(tr_list) < periodo:
            return 0
        
        avg_tr = np.mean(tr_list[-periodo:])
        avg_dm_plus = np.mean(dm_plus[-periodo:])
        avg_dm_minus = np.mean(dm_minus[-periodo:])
        
        if avg_tr == 0:
            return 0
        
        di_plus = (avg_dm_plus / avg_tr) * 100
        di_minus = (avg_dm_minus / avg_tr) * 100
        
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
        return dx
    
    @staticmethod
    def cci(df: pd.DataFrame, periodo: int = 20) -> float:
        """Commodity Channel Index"""
        if len(df) < periodo:
            return 0
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=periodo).mean().iloc[-1]
        mad = (tp.rolling(window=periodo).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)).iloc[-1]
        
        if mad == 0:
            return 0
        
        cci = (tp.iloc[-1] - sma_tp) / (0.015 * mad)
        return cci

    @staticmethod
    def momentum(dados: np.array, periodo: int = 10) -> float:
        """Momentum"""
        if len(dados) < periodo:
            return 0
        return dados[-1] - dados[-periodo]
    
    @staticmethod
    def rate_of_change(dados: np.array, periodo: int = 10) -> float:
        """Rate of Change"""
        if len(dados) < periodo:
            return 0
        return ((dados[-1] - dados[-periodo]) / dados[-periodo]) * 100

    @staticmethod
    def money_flow_index(df: pd.DataFrame, periodo: int = 14) -> float:
        """Money Flow Index"""
        if len(df) < periodo:
            return 50
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        
        positive_mf = []
        negative_mf = []
        
        for i in range(1, len(tp)):
            if tp.iloc[i] > tp.iloc[i-1]:
                positive_mf.append(mf.iloc[i])
                negative_mf.append(0)
            elif tp.iloc[i] < tp.iloc[i-1]:
                positive_mf.append(0)
                negative_mf.append(mf.iloc[i])
            else:
                positive_mf.append(0)
                negative_mf.append(0)
        
        if len(positive_mf) < periodo:
            return 50
        
        positive_sum = sum(positive_mf[-periodo:])
        negative_sum = sum(negative_mf[-periodo:])
        
        if negative_sum == 0:
            return 100
        
        mfr = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + mfr))
        return mfi

    @staticmethod
    def true_strength_index(df: pd.DataFrame, periodo1: int = 25, periodo2: int = 13) -> float:
        """True Strength Index"""
        if len(df) < max(periodo1, periodo2) + 1:
            return 0
        
        close = df['close'].values
        price_changes = np.diff(close)
        
        ema1_mom = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced.ema(price_changes, periodo1)
        ema2_mom = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced.ema([ema1_mom], periodo2)
        
        abs_changes = np.abs(price_changes)
        ema1_abs = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced.ema(abs_changes, periodo1)
        ema2_abs = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced.ema([ema1_abs], periodo2)
        
        if ema2_abs == 0:
            return 0
        
        tsi = (ema2_mom / ema2_abs) * 100
        return tsi

    @staticmethod
    def ultimate_oscillator(df: pd.DataFrame, periodo1: int = 7, periodo2: int = 14, periodo3: int = 28) -> float:
        """Ultimate Oscillator"""
        if len(df) < periodo3 + 1:
            return 50
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        bp_list = []
        tr_list = []
        
        for i in range(1, len(close)):
            bp = close[i] - min(low[i], close[i-1])
            tr = max(high[i], close[i-1]) - min(low[i], close[i-1])
            bp_list.append(bp)
            tr_list.append(tr)
        
        if len(bp_list) < periodo3:
            return 50
        
        bp1 = sum(bp_list[-periodo1:])
        tr1 = sum(tr_list[-periodo1:])
        avg1 = (bp1 / tr1) if tr1 != 0 else 0
        
        bp2 = sum(bp_list[-periodo2:])
        tr2 = sum(tr_list[-periodo2:])
        avg2 = (bp2 / tr2) if tr2 != 0 else 0
        
        bp3 = sum(bp_list[-periodo3:])
        tr3 = sum(tr_list[-periodo3:])
        avg3 = (bp3 / tr3) if tr3 != 0 else 0
        
        uo = ((4 * avg1) + (2 * avg2) + avg3) / 7 * 100
        return uo

    @staticmethod
    def parabolic_sar(df: pd.DataFrame, accel: float = 0.02, max_accel: float = 0.2) -> float:
        """Parabolic SAR"""
        if len(df) < 5:
            return df['close'].iloc[-1] if len(df) > 0 else 0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        trend = 1 if close[-1] > close[-2] else -1
        ep = high[-1] if trend == 1 else low[-1]
        sar = low[-2] if trend == 1 else high[-2]
        
        af = accel
        sar_new = sar + af * (ep - sar)
        
        return sar_new

    @staticmethod
    def awesome_oscillator(df: pd.DataFrame, periodo1: int = 5, periodo2: int = 34) -> float:
        """Awesome Oscillator"""
        if len(df) < periodo2:
            return 0
        
        median = (df['high'] + df['low']) / 2
        
        sma5 = median.rolling(window=periodo1).mean().iloc[-1]
        sma34 = median.rolling(window=periodo2).mean().iloc[-1]
        
        ao = sma5 - sma34
        return ao

print("✅ ARSENAL TÉCNICO COMPLETO V8 - ROYAL SUPREME ENHANCED CARREGADO!")
