#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 ENHANCED TECHNICAL ANALYSIS - ROYAL SUPREME ENHANCED 👑
💎 S/R + LTA/LTB + PULLBACK/THROWBACK + ELLIOTT + PRICE ACTION
🔥 ANÁLISE TÉCNICA AVANÇADA FUNCIONAL
🚀 NOVA FUNCIONALIDADE: S/R FLIP DETECTION PARA REDUZIR GALEs
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config_royal import ConfigRoyalSupremeEnhanced

class EnhancedTechnicalAnalysis:
    
    @staticmethod
    def detect_support_resistance_advanced(df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta suportes e resistências avançados"""
        if len(df) < ConfigRoyalSupremeEnhanced.SR_LOOKBACK_PERIODS:
            return {'support_levels': [], 'resistance_levels': [], 'score_call': 0, 'score_put': 0}
        
        highs = df['high'].values[-ConfigRoyalSupremeEnhanced.SR_LOOKBACK_PERIODS:]
        lows = df['low'].values[-ConfigRoyalSupremeEnhanced.SR_LOOKBACK_PERIODS:]
        closes = df['close'].values[-ConfigRoyalSupremeEnhanced.SR_LOOKBACK_PERIODS:]
        current_price = closes[-1]
        
        support_levels = []
        for i in range(2, len(lows) - 2):
            if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                
                touches = sum(1 for low in lows if abs(low - lows[i]) / lows[i] < ConfigRoyalSupremeEnhanced.SR_TOUCH_TOLERANCE)
                
                if touches >= ConfigRoyalSupremeEnhanced.SR_STRENGTH_MIN:
                    support_levels.append({
                        'level': lows[i],
                        'strength': touches,
                        'distance': abs(current_price - lows[i]) / current_price
                    })
        
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                
                touches = sum(1 for high in highs if abs(high - highs[i]) / highs[i] < ConfigRoyalSupremeEnhanced.SR_TOUCH_TOLERANCE)
                
                if touches >= ConfigRoyalSupremeEnhanced.SR_STRENGTH_MIN:
                    resistance_levels.append({
                        'level': highs[i],
                        'strength': touches,
                        'distance': abs(current_price - highs[i]) / current_price
                    })
        
        score_call = 0
        score_put = 0
        
        for support in support_levels:
            if support['distance'] < 0.01:
                score_call += support['strength'] * 15
        
        for resistance in resistance_levels:
            if resistance['distance'] < 0.01:
                score_put += resistance['strength'] * 15
        
        return {
            'support_levels': sorted(support_levels, key=lambda x: x['distance'])[:3],
            'resistance_levels': sorted(resistance_levels, key=lambda x: x['distance'])[:3],
            'score_call': min(score_call, 60),
            'score_put': min(score_put, 60),
            'nearest_support': min(support_levels, key=lambda x: x['distance']) if support_levels else None,
            'nearest_resistance': min(resistance_levels, key=lambda x: x['distance']) if resistance_levels else None
        }
    
    @staticmethod
    def detect_sr_flip_supreme(df: pd.DataFrame) -> Dict[str, Any]:
        """🚀 NOVO: Detecta S/R Flip para timing perfeito e redução de GALEs"""
        if len(df) < 100:
            return {'flip_detected': False, 'flip_type': None, 'score_call': 0, 'score_put': 0}
        
        # Analisar dados das últimas 100 velas
        recent_data = df.tail(100)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        closes = recent_data['close'].values
        current_price = closes[-1]
        
        # Detectar níveis S/R históricos
        historical_sr_levels = []
        
        # Buscar suportes que viraram resistência
        for i in range(5, len(lows) - 10):
            # Verificar se era suporte (múltiplos toques)
            level = lows[i]
            touches_as_support = 0
            touches_as_resistance = 0
            
            # Contar toques como suporte (antes do rompimento)
            for j in range(max(0, i-20), i):
                if abs(lows[j] - level) / level < 0.002:  # Tolerância 0.2%
                    touches_as_support += 1
            
            # Verificar se foi rompido para baixo
            if touches_as_support >= 2:  # Era suporte válido
                # Verificar rompimento
                broke_down = False
                break_point = None
                
                for j in range(i, min(len(closes), i + 15)):
                    if closes[j] < level * 0.997:  # Rompeu suporte
                        broke_down = True
                        break_point = j
                        break
                
                if broke_down and break_point:
                    # Verificar se agora está testando como resistência
                    for j in range(break_point, len(closes)):
                        if abs(highs[j] - level) / level < 0.002:  # Testou como resistência
                            touches_as_resistance += 1
                    
                    # FLIP DETECTADO: Suporte virou Resistência
                    if touches_as_resistance >= 1:
                        distance = abs(current_price - level) / current_price
                        
                        historical_sr_levels.append({
                            'level': level,
                            'type': 'SUPPORT_TO_RESISTANCE',
                            'original_touches': touches_as_support,
                            'new_touches': touches_as_resistance,
                            'strength': touches_as_support + touches_as_resistance,
                            'distance': distance
                        })
        
        # Buscar resistências que viraram suporte  
        for i in range(5, len(highs) - 10):
            # Verificar se era resistência (múltiplos toques)
            level = highs[i]
            touches_as_resistance = 0
            touches_as_support = 0
            
            # Contar toques como resistência (antes do rompimento)
            for j in range(max(0, i-20), i):
                if abs(highs[j] - level) / level < 0.002:  # Tolerância 0.2%
                    touches_as_resistance += 1
            
            # Verificar se foi rompido para cima
            if touches_as_resistance >= 2:  # Era resistência válida
                # Verificar rompimento
                broke_up = False
                break_point = None
                
                for j in range(i, min(len(closes), i + 15)):
                    if closes[j] > level * 1.003:  # Rompeu resistência
                        broke_up = True
                        break_point = j
                        break
                
                if broke_up and break_point:
                    # Verificar se agora está testando como suporte
                    for j in range(break_point, len(closes)):
                        if abs(lows[j] - level) / level < 0.002:  # Testou como suporte
                            touches_as_support += 1
                    
                    # FLIP DETECTADO: Resistência virou Suporte
                    if touches_as_support >= 1:
                        distance = abs(current_price - level) / current_price
                        
                        historical_sr_levels.append({
                            'level': level,
                            'type': 'RESISTANCE_TO_SUPPORT',
                            'original_touches': touches_as_resistance,
                            'new_touches': touches_as_support,
                            'strength': touches_as_resistance + touches_as_support,
                            'distance': distance
                        })
        
        # Analisar flips mais próximos
        if not historical_sr_levels:
            return {'flip_detected': False, 'flip_type': None, 'score_call': 0, 'score_put': 0}
        
        # Ordenar por proximidade
        historical_sr_levels.sort(key=lambda x: x['distance'])
        nearest_flip = historical_sr_levels[0]
        
        score_call = 0
        score_put = 0
        flip_detected = False
        flip_analysis = {}
        
        # Calcular timing score baseado na proximidade e força
        if nearest_flip['distance'] < 0.015:  # Dentro de 1.5%
            strength_multiplier = min(nearest_flip['strength'] * 10, 60)
            proximity_bonus = max(0, (0.015 - nearest_flip['distance']) * 1000)
            
            flip_detected = True
            flip_analysis = {
                'level': nearest_flip['level'],
                'type': nearest_flip['type'],
                'strength': nearest_flip['strength'],
                'distance_pct': nearest_flip['distance'] * 100,
                'timing_score': strength_multiplier + proximity_bonus
            }
            
            # Determinar direção do trade baseado no flip
            if nearest_flip['type'] == 'RESISTANCE_TO_SUPPORT':
                # Era resistência, agora é suporte → CALL se está próximo
                if current_price >= nearest_flip['level'] * 0.998:  # Acima do novo suporte
                    score_call = strength_multiplier + proximity_bonus
                    flip_analysis['trade_direction'] = 'CALL'
                    flip_analysis['reason'] = f"Ex-resistência virando suporte forte ({nearest_flip['strength']} toques)"
                
            elif nearest_flip['type'] == 'SUPPORT_TO_RESISTANCE':
                # Era suporte, agora é resistência → PUT se está próximo
                if current_price <= nearest_flip['level'] * 1.002:  # Abaixo da nova resistência
                    score_put = strength_multiplier + proximity_bonus
                    flip_analysis['trade_direction'] = 'PUT'
                    flip_analysis['reason'] = f"Ex-suporte virando resistência forte ({nearest_flip['strength']} toques)"
        
        return {
            'flip_detected': flip_detected,
            'flip_type': nearest_flip['type'] if flip_detected else None,
            'flip_analysis': flip_analysis if flip_detected else {},
            'score_call': min(score_call, 80),  # Máximo 80 pontos
            'score_put': min(score_put, 80),
            'total_flips_found': len(historical_sr_levels),
            'all_flips': historical_sr_levels[:3]  # Top 3 mais próximos
        }
    
    @staticmethod
    def detect_trend_lines(df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta linhas de tendência LTA/LTB"""
        if len(df) < ConfigRoyalSupremeEnhanced.TRENDLINE_LOOKBACK:
            return {'lta': None, 'ltb': None, 'score_call': 0, 'score_put': 0}
        
        highs = df['high'].values[-ConfigRoyalSupremeEnhanced.TRENDLINE_LOOKBACK:]
        lows = df['low'].values[-ConfigRoyalSupremeEnhanced.TRENDLINE_LOOKBACK:]
        current_price = df['close'].iloc[-1]
        
        # Detectar LTA (Linha de Tendência de Alta)
        lta_points = []
        for i in range(2, len(lows) - 2):
            if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                lta_points.append((i, lows[i]))
        
        lta = None
        if len(lta_points) >= ConfigRoyalSupremeEnhanced.TRENDLINE_MIN_TOUCHES:
            for i in range(len(lta_points) - 1):
                for j in range(i + 1, len(lta_points)):
                    if lta_points[j][1] > lta_points[i][1]:
                        slope = (lta_points[j][1] - lta_points[i][1]) / (lta_points[j][0] - lta_points[i][0])
                        current_level = lta_points[j][1] + slope * (len(lows) - 1 - lta_points[j][0])
                        
                        if abs(current_price - current_level) / current_price < ConfigRoyalSupremeEnhanced.TRENDLINE_TOLERANCE:
                            lta = {
                                'level': current_level,
                                'slope': slope,
                                'strength': 2,
                                'type': 'LTA'
                            }
        
        # Detectar LTB (Linha de Tendência de Baixa)
        ltb_points = []
        for i in range(2, len(highs) - 2):
            if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                ltb_points.append((i, highs[i]))
        
        ltb = None
        if len(ltb_points) >= ConfigRoyalSupremeEnhanced.TRENDLINE_MIN_TOUCHES:
            for i in range(len(ltb_points) - 1):
                for j in range(i + 1, len(ltb_points)):
                    if ltb_points[j][1] < ltb_points[i][1]:
                        slope = (ltb_points[j][1] - ltb_points[i][1]) / (ltb_points[j][0] - ltb_points[i][0])
                        current_level = ltb_points[j][1] + slope * (len(highs) - 1 - ltb_points[j][0])
                        
                        if abs(current_price - current_level) / current_price < ConfigRoyalSupremeEnhanced.TRENDLINE_TOLERANCE:
                            ltb = {
                                'level': current_level,
                                'slope': slope,
                                'strength': 2,
                                'type': 'LTB'
                            }
        
        score_call = 0
        score_put = 0
        
        if lta and current_price >= lta['level'] * 0.998:
            score_call += 25
        
        if ltb and current_price <= ltb['level'] * 1.002:
            score_put += 25
        
        return {
            'lta': lta,
            'ltb': ltb,
            'score_call': score_call,
            'score_put': score_put
        }
    
    @staticmethod
    def detect_pullback_throwback(df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta pullbacks e throwbacks"""
        if len(df) < 20:
            return {'pullback': None, 'throwback': None, 'score_call': 0, 'score_put': 0}
        
        closes = df['close'].values[-20:]
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]
        
        score_call = 0
        score_put = 0
        pullback = None
        throwback = None
        
        # Detectar PULLBACK (retração em tendência de alta)
        recent_high = np.max(highs[-10:])
        recent_low = np.min(lows[-5:])
        current_price = closes[-1]
        
        if recent_high > closes[-10]:
            retrace_pct = (recent_high - current_price) / (recent_high - recent_low) if recent_high != recent_low else 0
            
            if (ConfigRoyalSupremeEnhanced.PULLBACK_MIN_RETRACE <= retrace_pct <= 
                ConfigRoyalSupremeEnhanced.PULLBACK_MAX_RETRACE):
                
                # Verificar se está retomando a tendência
                if closes[-1] > closes[-2] and closes[-2] > closes[-3]:
                    pullback = {
                        'type': 'PULLBACK',
                        'retrace_pct': retrace_pct,
                        'strength': 'STRONG' if retrace_pct > 0.5 else 'MEDIUM'
                    }
                    score_call += 30
        
        # Detectar THROWBACK (retração em tendência de baixa)
        recent_low_tb = np.min(lows[-10:])
        recent_high_tb = np.max(highs[-5:])
        
        if recent_low_tb < closes[-10]:
            retrace_pct = (current_price - recent_low_tb) / (recent_high_tb - recent_low_tb) if recent_high_tb != recent_low_tb else 0
            
            if (ConfigRoyalSupremeEnhanced.PULLBACK_MIN_RETRACE <= retrace_pct <= 
                ConfigRoyalSupremeEnhanced.PULLBACK_MAX_RETRACE):
                
                # Verificar se está retomando a tendência
                if closes[-1] < closes[-2] and closes[-2] < closes[-3]:
                    throwback = {
                        'type': 'THROWBACK',
                        'retrace_pct': retrace_pct,
                        'strength': 'STRONG' if retrace_pct > 0.5 else 'MEDIUM'
                    }
                    score_put += 30
        
        return {
            'pullback': pullback,
            'throwback': throwback,
            'score_call': score_call,
            'score_put': score_put
        }
    
    @staticmethod
    def detect_elliott_waves_basic(df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta padrões básicos de Elliott Waves"""
        if len(df) < 30:
            return {'wave_pattern': None, 'score_call': 0, 'score_put': 0}
        
        closes = df['close'].values[-30:]
        
        score_call = 0
        score_put = 0
        wave_pattern = None
        
        # Detectar pontos pivôs
        pivot_points = []
        for i in range(2, len(closes) - 2):
            try:
                if (closes[i] > closes[i-1] and closes[i] > closes[i-2] and 
                    closes[i] > closes[i+1] and closes[i] > closes[i+2]):
                    pivot_points.append(('HIGH', i, closes[i]))
                elif (closes[i] < closes[i-1] and closes[i] < closes[i-2] and 
                      closes[i] < closes[i+1] and closes[i] < closes[i+2]):
                    pivot_points.append(('LOW', i, closes[i]))
            except (IndexError, ValueError):
                continue
        
        # Analisar padrão de 5 ondas
        if len(pivot_points) >= 5:
            last_5 = pivot_points[-5:]
            
            # Padrão bullish: LOW-HIGH-LOW-HIGH-LOW
            if (last_5[0][0] == 'LOW' and last_5[1][0] == 'HIGH' and 
                last_5[2][0] == 'LOW' and last_5[3][0] == 'HIGH' and 
                last_5[4][0] == 'LOW'):
                
                # Verificar se onda 5 é menor que onda 3
                if last_5[4][2] < last_5[2][2]:
                    wave_pattern = {
                        'type': 'ELLIOTT_WAVE_5_BULLISH',
                        'direction': 'CALL',
                        'confidence': 'MEDIUM'
                    }
                    score_call += 35
            
            # Padrão bearish: HIGH-LOW-HIGH-LOW-HIGH
            elif (last_5[0][0] == 'HIGH' and last_5[1][0] == 'LOW' and 
                  last_5[2][0] == 'HIGH' and last_5[3][0] == 'LOW' and 
                  last_5[4][0] == 'HIGH'):
                
                # Verificar se onda 5 é maior que onda 3
                if last_5[4][2] > last_5[2][2]:
                    wave_pattern = {
                        'type': 'ELLIOTT_WAVE_5_BEARISH',
                        'direction': 'PUT',
                        'confidence': 'MEDIUM'
                    }
                    score_put += 35
        
        return {
            'wave_pattern': wave_pattern,
            'score_call': score_call,
            'score_put': score_put,
            'pivot_points': len(pivot_points)
        }

class PriceActionMasterRoyalSupremeEnhanced:
    
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Detecta padrões de candlesticks"""
        if len(df) < 3:
            return {'patterns': [], 'score_call': 0, 'score_put': 0}
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        o, h, l, c = current['open'], current['high'], current['low'], current['close']
        po, ph, pl, pc = prev['open'], prev['high'], prev['low'], prev['close']
        
        patterns = []
        score_call = 0
        score_put = 0
        
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        
        # DOJI
        if body <= (h - l) * 0.1:
            patterns.append("Doji")
            score_call += 10
            score_put += 10
        
        # HAMMER
        if (lower_shadow >= body * 2 and upper_shadow <= body * 0.1 and 
            c > o and l < pl):
            patterns.append("Hammer")
            score_call += 35
        
        # SHOOTING STAR
        if (upper_shadow >= body * 2 and lower_shadow <= body * 0.1 and 
            c < o and h > ph):
            patterns.append("Shooting Star")
            score_put += 35
        
        # BULLISH ENGULFING
        if (c > o and pc < po and c > po and o < pc and body > abs(pc - po)):
            patterns.append("Bullish Engulfing")
            score_call += 40
        
        # BEARISH ENGULFING
        if (c < o and pc > po and c < po and o > pc and body > abs(pc - po)):
            patterns.append("Bearish Engulfing")
            score_put += 40
        
        # PIN BAR BULLISH - CORRIGIDO
        if (lower_shadow >= (h - l) * 0.6 and upper_shadow <= (h - l) * 0.2 and 
            c > (h + l) / 2):  # CORREÇÃO: Close na metade superior da vela
            patterns.append("Pin Bar Bullish")
            score_call += 30
        
        # PIN BAR BEARISH - CORRIGIDO  
        if (upper_shadow >= (h - l) * 0.6 and lower_shadow <= (h - l) * 0.2 and 
            c < (h + l) / 2):  # CORREÇÃO: Close na metade inferior da vela
            patterns.append("Pin Bar Bearish")
            score_put += 30
        
        return {
            'patterns': patterns,
            'score_call': score_call,
            'score_put': score_put,
            'total_patterns': len(patterns)
        }
    
    @staticmethod
    def analyze_complete_price_action_enhanced(df: pd.DataFrame) -> Dict[str, Any]:
        """🚀 VERSÃO UPGRADADA: Análise completa com S/R Flip Detection"""
        candlestick = PriceActionMasterRoyalSupremeEnhanced.detect_candlestick_patterns(df)
        
        sr_analysis = EnhancedTechnicalAnalysis.detect_support_resistance_advanced(df)
        
        # 🚀 NOVA FUNCIONALIDADE: S/R Flip Detection
        flip_analysis = EnhancedTechnicalAnalysis.detect_sr_flip_supreme(df)
        
        trendlines = EnhancedTechnicalAnalysis.detect_trend_lines(df)
        pullback_throwback = EnhancedTechnicalAnalysis.detect_pullback_throwback(df)
        elliott = EnhancedTechnicalAnalysis.detect_elliott_waves_basic(df)
        
        # 🎯 SCORE TOTAL COM S/R FLIP BOOST
        total_score_call = (candlestick['score_call'] + 
                           sr_analysis['score_call'] + 
                           flip_analysis['score_call'] +  # 🚀 NOVO BOOST
                           trendlines['score_call'] +
                           pullback_throwback['score_call'] +
                           elliott['score_call'])
        
        total_score_put = (candlestick['score_put'] + 
                          sr_analysis['score_put'] + 
                          flip_analysis['score_put'] +   # 🚀 NOVO BOOST
                          trendlines['score_put'] +
                          pullback_throwback['score_put'] +
                          elliott['score_put'])
        
        motivos_call = []
        motivos_put = []
        
        # 🚀 MOTIVOS S/R FLIP (PRIORIDADE MÁXIMA)
        if flip_analysis['flip_detected']:
            flip_info = flip_analysis['flip_analysis']
            if flip_info.get('trade_direction') == 'CALL':
                motivos_call.append(f"🔄 S/R Flip: {flip_info['reason']}")
            elif flip_info.get('trade_direction') == 'PUT':
                motivos_put.append(f"🔄 S/R Flip: {flip_info['reason']}")
        
        # Compilar motivos CALL
        if candlestick['patterns']:
            for pattern in candlestick['patterns']:
                if any(x in pattern for x in ['Hammer', 'Bullish', 'Pin Bar Bullish']):
                    motivos_call.append(f"Candlestick: {pattern}")
                elif any(x in pattern for x in ['Shooting Star', 'Bearish', 'Pin Bar Bearish']):
                    motivos_put.append(f"Candlestick: {pattern}")
                else:
                    motivos_call.append(f"Candlestick: {pattern}")
                    motivos_put.append(f"Candlestick: {pattern}")
        
        if sr_analysis['score_call'] > 0:
            motivos_call.append("S/R: Suporte Forte Detectado")
        if sr_analysis['score_put'] > 0:
            motivos_put.append("S/R: Resistência Forte Detectada")
        
        if trendlines['lta']:
            motivos_call.append("LTA: Linha Tendência Alta")
        if trendlines['ltb']:
            motivos_put.append("LTB: Linha Tendência Baixa")
        
        if pullback_throwback['pullback']:
            motivos_call.append(f"Pullback: {pullback_throwback['pullback']['strength']}")
        if pullback_throwback['throwback']:
            motivos_put.append(f"Throwback: {pullback_throwback['throwback']['strength']}")
        
        if elliott['wave_pattern']:
            if elliott['wave_pattern']['direction'] == 'CALL':
                motivos_call.append(f"Elliott: {elliott['wave_pattern']['type']}")
            else:
                motivos_put.append(f"Elliott: {elliott['wave_pattern']['type']}")
        
        return {
            'price_action_score_call': total_score_call,
            'price_action_score_put': total_score_put,
            'price_action_motivos_call': motivos_call,
            'price_action_motivos_put': motivos_put,
            'candlestick_patterns': candlestick['patterns'],
            'support_levels': sr_analysis['support_levels'],
            'resistance_levels': sr_analysis['resistance_levels'],
            'lta': trendlines['lta'],
            'ltb': trendlines['ltb'],
            'pullback': pullback_throwback['pullback'],
            'throwback': pullback_throwback['throwback'],
            'elliott_pattern': elliott['wave_pattern'],
            # 🚀 NOVOS DADOS DO S/R FLIP
            'sr_flip_detected': flip_analysis['flip_detected'],
            'sr_flip_type': flip_analysis['flip_type'],
            'sr_flip_analysis': flip_analysis.get('flip_analysis', {}),
            'sr_flip_score_call': flip_analysis['score_call'],
            'sr_flip_score_put': flip_analysis['score_put'],
            'total_flips_found': flip_analysis.get('total_flips_found', 0),
            'total_enhanced_signals': len(motivos_call) + len(motivos_put)
        }

print("✅ ENHANCED TECHNICAL ANALYSIS - ROYAL SUPREME ENHANCED CARREGADO!")
print("🚀 NOVA FUNCIONALIDADE: S/R FLIP DETECTION PARA REDUZIR GALEs ATIVADA!")
print("🔧 PIN BAR LOGIC CORRIGIDA - VERSÃO FINAL!")