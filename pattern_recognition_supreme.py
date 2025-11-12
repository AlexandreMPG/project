#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 PATTERN RECOGNITION SUPREME - SISTEMA AVANÇADO DE PADRÕES 👑
💎 50+ PADRÕES GRÁFICOS + ANÁLISE CONTEXTUAL + MACRO/MICRO TENDÊNCIAS
🔥 BASEADO NAS IMAGENS FORNECIDAS + TRADE WITH TREND + IA CONTEXTUAL
🚀 EVITA LOSSES POR PADRÕES DE INDECISÃO + DETECTA FASES DE MERCADO
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

class TrendDirection(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

class MarketPhase(Enum):
    IMPULSE = "IMPULSE"          # Movimento forte na direção da tendência
    CORRECTION = "CORRECTION"    # Retração natural (não operar contra)
    RESUMPTION = "RESUMPTION"    # Retomada da tendência (melhor momento)
    EXHAUSTION = "EXHAUSTION"    # Fim do movimento (cuidado)
    CONSOLIDATION = "CONSOLIDATION" # Lateralização

class PatternStrength(Enum):
    VERY_STRONG = 60
    STRONG = 45
    MEDIUM = 30
    WEAK = 15
    VERY_WEAK = 5
    DANGEROUS = -50  # Padrões que devem ser evitados

class TrendAnalysisSupreme:
    """Sistema avançado de análise de tendências macro e micro"""
    
    def analyze_macro_micro_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa tendências macro (1000 velas) e micro (50 velas) para confluências"""
        
        # MACRO ANÁLISE (máximo de velas disponíveis, idealmente 1000)
        macro_period = min(len(df), 1000)
        micro_period = min(len(df), 50)
        
        if len(df) < 20:
            return {
                'macro_trend': 'NEUTRAL',
                'micro_trend': 'NEUTRAL',
                'macro_micro_confluence': False,
                'confluence_direction': None,
                'confluence_strength': 0
            }
        
        # ANÁLISE MACRO (máximo disponível)
        macro_data = df.tail(macro_period)
        macro_trend = self._analyze_trend_direction(macro_data, 'MACRO')
        
        # ANÁLISE MICRO (últimas 50 velas)
        micro_data = df.tail(micro_period)
        micro_trend = self._analyze_trend_direction(micro_data, 'MICRO')
        
        # DETECTAR CONFLUÊNCIA MACRO/MICRO
        confluence_analysis = self._detect_macro_micro_confluence(macro_trend, micro_trend)
        
        # ANÁLISE DE LTA/LTB EM MÚLTIPLOS TIMEFRAMES
        lta_ltb_analysis = self._analyze_multi_timeframe_trendlines(df)
        
        return {
            'macro_trend': macro_trend['direction'],
            'micro_trend': micro_trend['direction'],
            'macro_strength': macro_trend['strength'],
            'micro_strength': micro_trend['strength'],
            'macro_micro_confluence': confluence_analysis['has_confluence'],
            'confluence_direction': confluence_analysis['direction'],
            'confluence_strength': confluence_analysis['strength'],
            'lta_macro': lta_ltb_analysis['lta_macro'],
            'ltb_macro': lta_ltb_analysis['ltb_macro'],
            'lta_micro': lta_ltb_analysis['lta_micro'],
            'ltb_micro': lta_ltb_analysis['ltb_micro'],
            'trendline_confluence': lta_ltb_analysis['confluence']
        }
    
    def _analyze_trend_direction(self, df: pd.DataFrame, timeframe_type: str) -> Dict[str, Any]:
        """Analisa direção e força da tendência"""
        
        if len(df) < 10:
            return {'direction': 'NEUTRAL', 'strength': 0}
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Médias móveis para tendência
        if len(closes) >= 20:
            ma20 = np.mean(closes[-20:])
            ma50 = np.mean(closes[-min(50, len(closes)):]) if len(closes) >= 50 else ma20
            ma200 = np.mean(closes[-min(200, len(closes)):]) if len(closes) >= 200 else ma50
        else:
            ma20 = ma50 = ma200 = np.mean(closes)
        
        current_price = closes[-1]
        
        # ANÁLISE DE TENDÊNCIA BASEADA EM MÚLTIPLOS FATORES
        trend_signals = []
        
        # 1. Posição em relação às médias
        if current_price > ma20 > ma50 > ma200:
            trend_signals.append(('STRONG_BULLISH', 4))
        elif current_price > ma20 > ma50:
            trend_signals.append(('BULLISH', 3))
        elif current_price > ma20:
            trend_signals.append(('BULLISH', 2))
        elif current_price < ma20 < ma50 < ma200:
            trend_signals.append(('STRONG_BEARISH', 4))
        elif current_price < ma20 < ma50:
            trend_signals.append(('BEARISH', 3))
        elif current_price < ma20:
            trend_signals.append(('BEARISH', 2))
        else:
            trend_signals.append(('NEUTRAL', 1))
        
        # 2. Sequência de máximas e mínimas
        if len(closes) >= 10:
            recent_highs = []
            recent_lows = []
            
            for i in range(5, len(closes)):
                if i < len(highs)-1:
                    if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                        recent_highs.append(highs[i])
                if i < len(lows)-1:
                    if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                        recent_lows.append(lows[i])
            
            if len(recent_highs) >= 2:
                if recent_highs[-1] > recent_highs[-2]:
                    trend_signals.append(('BULLISH', 2))
                else:
                    trend_signals.append(('BEARISH', 1))
            
            if len(recent_lows) >= 2:
                if recent_lows[-1] > recent_lows[-2]:
                    trend_signals.append(('BULLISH', 2))
                else:
                    trend_signals.append(('BEARISH', 1))
        
        # 3. Momentum de preços
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
            if momentum > 1.0:
                trend_signals.append(('STRONG_BULLISH', 3))
            elif momentum > 0.3:
                trend_signals.append(('BULLISH', 2))
            elif momentum < -1.0:
                trend_signals.append(('STRONG_BEARISH', 3))
            elif momentum < -0.3:
                trend_signals.append(('BEARISH', 2))
        
        # CALCULAR TENDÊNCIA FINAL
        bullish_score = sum(weight for direction, weight in trend_signals if 'BULLISH' in direction)
        bearish_score = sum(weight for direction, weight in trend_signals if 'BEARISH' in direction)
        
        if bullish_score > bearish_score + 3:
            if bullish_score >= 8:
                trend_direction = 'STRONG_BULLISH'
                strength = min(bullish_score * 10, 100)
            else:
                trend_direction = 'BULLISH'
                strength = min(bullish_score * 8, 80)
        elif bearish_score > bullish_score + 3:
            if bearish_score >= 8:
                trend_direction = 'STRONG_BEARISH'
                strength = min(bearish_score * 10, 100)
            else:
                trend_direction = 'BEARISH'
                strength = min(bearish_score * 8, 80)
        else:
            trend_direction = 'NEUTRAL'
            strength = 20
        
        return {
            'direction': trend_direction,
            'strength': strength,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200
        }
    
    def _detect_macro_micro_confluence(self, macro_trend: Dict, micro_trend: Dict) -> Dict[str, Any]:
        """Detecta confluência entre tendências macro e micro"""
        
        macro_dir = macro_trend['direction']
        micro_dir = micro_trend['direction']
        
        # CONFLUÊNCIA BULLISH FORTE
        if ('BULLISH' in macro_dir and 'BULLISH' in micro_dir):
            if 'STRONG' in macro_dir and 'STRONG' in micro_dir:
                return {
                    'has_confluence': True,
                    'direction': 'BULLISH',
                    'strength': 70,  # Score boost muito forte
                    'type': 'STRONG_BULL_CONFLUENCE'
                }
            else:
                return {
                    'has_confluence': True,
                    'direction': 'BULLISH',
                    'strength': 45,  # Score boost moderado
                    'type': 'BULL_CONFLUENCE'
                }
        
        # CONFLUÊNCIA BEARISH FORTE
        elif ('BEARISH' in macro_dir and 'BEARISH' in micro_dir):
            if 'STRONG' in macro_dir and 'STRONG' in micro_dir:
                return {
                    'has_confluence': True,
                    'direction': 'BEARISH',
                    'strength': 70,  # Score boost muito forte
                    'type': 'STRONG_BEAR_CONFLUENCE'
                }
            else:
                return {
                    'has_confluence': True,
                    'direction': 'BEARISH',
                    'strength': 45,  # Score boost moderado
                    'type': 'BEAR_CONFLUENCE'
                }
        
        # SEM CONFLUÊNCIA OU CONFLITO
        else:
            return {
                'has_confluence': False,
                'direction': None,
                'strength': 0,
                'type': 'NO_CONFLUENCE'
            }
    
    def _analyze_multi_timeframe_trendlines(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa LTA/LTB em múltiplos timeframes"""
        
        result = {
            'lta_macro': None,
            'ltb_macro': None,
            'lta_micro': None,
            'ltb_micro': None,
            'confluence': False
        }
        
        if len(df) < 50:
            return result
        
        # MACRO LTA/LTB (200+ velas se disponível)
        macro_period = min(len(df), 200)
        if macro_period >= 50:
            macro_data = df.tail(macro_period)
            result['lta_macro'] = self._detect_trendline(macro_data, 'LTA')
            result['ltb_macro'] = self._detect_trendline(macro_data, 'LTB')
        
        # MICRO LTA/LTB (50 velas)
        micro_data = df.tail(50)
        result['lta_micro'] = self._detect_trendline(micro_data, 'LTA')
        result['ltb_micro'] = self._detect_trendline(micro_data, 'LTB')
        
        # DETECTAR CONFLUÊNCIA DE TRENDLINES
        if ((result['lta_macro'] and result['lta_micro']) or 
            (result['ltb_macro'] and result['ltb_micro'])):
            result['confluence'] = True
        
        return result
    
    def _detect_trendline(self, df: pd.DataFrame, line_type: str) -> Optional[Dict]:
        """Detecta linha de tendência específica"""
        
        if len(df) < 20:
            return None
        
        if line_type == 'LTA':
            # Detectar LTA (suportes crescentes)
            lows = df['low'].values
            points = []
            
            for i in range(2, len(lows) - 2):
                if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                    lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                    points.append((i, lows[i]))
            
            # Encontrar linha de tendência ascendente
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    for j in range(i + 1, len(points)):
                        if points[j][1] > points[i][1]:  # Ascendente
                            return {
                                'type': 'LTA',
                                'strength': 'STRONG' if len(points) >= 3 else 'MEDIUM',
                                'points': len(points)
                            }
        
        elif line_type == 'LTB':
            # Detectar LTB (resistências decrescentes)
            highs = df['high'].values
            points = []
            
            for i in range(2, len(highs) - 2):
                if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                    highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                    points.append((i, highs[i]))
            
            # Encontrar linha de tendência descendente
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    for j in range(i + 1, len(points)):
                        if points[j][1] < points[i][1]:  # Descendente
                            return {
                                'type': 'LTB',
                                'strength': 'STRONG' if len(points) >= 3 else 'MEDIUM',
                                'points': len(points)
                            }
        
        return None
    
    def analyze_momentum_phase(self, df: pd.DataFrame, movimento_1min: float) -> Dict[str, Any]:
        """🚀 RESOLVE O PROBLEMA: Analisa em que fase do momentum o mercado está"""
        
        if len(df) < 10:
            return {'phase': 'NORMAL', 'strength': 0}
        
        closes = df['close'].values
        
        # Calcular médias para contexto
        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else ma5
        current_price = closes[-1]
        
        # Analisar movimentos recentes
        recent_moves = []
        for i in range(1, min(6, len(closes))):
            if len(closes) > i:
                move = ((closes[-i] - closes[-i-1]) / closes[-i-1]) * 100
                recent_moves.append(move)
        
        avg_recent_move = np.mean(recent_moves) if recent_moves else 0
        
        # 🔥 DETECTAR FASE DO MOMENTUM (CRÍTICO PARA EVITAR LOSSES)
        
        # IMPULSE: Movimento forte na direção da tendência
        if abs(movimento_1min) > 0.15 and abs(avg_recent_move) > 0.08:
            return {
                'phase': 'IMPULSE',
                'strength': min(abs(movimento_1min) * 100, 80),
                'direction': 'BULLISH' if movimento_1min > 0 else 'BEARISH'
            }
        
        # CORRECTION: Movimento contra a tendência principal (🚫 NÃO OPERAR CONTRA)
        if current_price > ma20:  # Tendência de alta
            if movimento_1min < -0.05:  # Movimento de baixa (correção)
                return {
                    'phase': 'CORRECTION',
                    'strength': abs(movimento_1min) * 100,
                    'direction': 'BEARISH_CORRECTION'
                }
        elif current_price < ma20:  # Tendência de baixa
            if movimento_1min > 0.05:  # Movimento de alta (correção)
                return {
                    'phase': 'CORRECTION',
                    'strength': abs(movimento_1min) * 100,
                    'direction': 'BULLISH_CORRECTION'
                }
        
        # RESUMPTION: Retomada da tendência após correção (⚡ MELHOR MOMENTO)
        if len(recent_moves) >= 3:
            # Verificar se há padrão de correção seguida de retomada
            if current_price > ma20:  # Tendência de alta
                if recent_moves[0] < 0 and movimento_1min > 0.03:  # Correção + retomada
                    return {
                        'phase': 'RESUMPTION',
                        'strength': movimento_1min * 150,  # Boost forte
                        'direction': 'BULLISH_RESUMPTION'
                    }
            elif current_price < ma20:  # Tendência de baixa
                if recent_moves[0] > 0 and movimento_1min < -0.03:  # Correção + retomada
                    return {
                        'phase': 'RESUMPTION',
                        'strength': abs(movimento_1min) * 150,  # Boost forte
                        'direction': 'BEARISH_RESUMPTION'
                    }
        
        # NORMAL: Movimento normal sem características especiais
        return {
            'phase': 'NORMAL',
            'strength': abs(movimento_1min) * 50,
            'direction': 'NEUTRAL'
        }

class PatternRecognitionSupreme:
    """Sistema avançado de reconhecimento de padrões baseado nas imagens fornecidas"""
    
    def __init__(self):
        # Inicializar padrões conhecidos com suas características
        self.pattern_database = self._initialize_pattern_database()
        self.dangerous_patterns = [
            'DOJI', 'SPINNING_TOP', 'INDECISION_CANDLE', 
            'WEAK_HAMMER', 'WEAK_SHOOTING_STAR'
        ]
    
    def _initialize_pattern_database(self) -> Dict[str, Dict]:
        """Inicializa base de dados com todos os padrões das imagens"""
        
        return {
            # PADRÕES BULLISH FORTES (da imagem 1 e 5)
            'HAMMER': {
                'strength': PatternStrength.STRONG,
                'context_required': 'DOWNTREND',
                'description': 'Martelo em tendência de baixa'
            },
            'BULLISH_ENGULFING': {
                'strength': PatternStrength.VERY_STRONG,
                'context_required': 'DOWNTREND',
                'description': 'Engolfo bullish forte'
            },
            'MORNING_STAR': {
                'strength': PatternStrength.VERY_STRONG,
                'context_required': 'DOWNTREND',
                'description': 'Estrela da manhã completa'
            },
            'PIERCING_PATTERN': {
                'strength': PatternStrength.STRONG,
                'context_required': 'DOWNTREND',
                'description': 'Padrão perfurante'
            },
            'BULLISH_HARAMI': {
                'strength': PatternStrength.MEDIUM,
                'context_required': 'DOWNTREND',
                'description': 'Harami bullish'
            },
            'THREE_WHITE_SOLDIERS': {
                'strength': PatternStrength.VERY_STRONG,
                'context_required': 'ANY',
                'description': 'Três soldados brancos'
            },
            'INVERTED_HAMMER': {
                'strength': PatternStrength.MEDIUM,
                'context_required': 'DOWNTREND',
                'description': 'Martelo invertido'
            },
            'TWEEZER_BOTTOM': {
                'strength': PatternStrength.STRONG,
                'context_required': 'DOWNTREND',
                'description': 'Tweezer Bottom'
            },
            'BULLISH_MARUBOZU': {
                'strength': PatternStrength.STRONG,
                'context_required': 'ANY',
                'description': 'Marubozu bullish'
            },
            'DRAGONFLY_DOJI': {
                'strength': PatternStrength.MEDIUM,
                'context_required': 'DOWNTREND',
                'description': 'Doji libélula'
            },
            'BULLISH_SPINNING_TOP': {
                'strength': PatternStrength.WEAK,
                'context_required': 'DOWNTREND',
                'description': 'Spinning top bullish'
            },
            'RISING_THREE_METHODS': {
                'strength': PatternStrength.MEDIUM,
                'context_required': 'UPTREND',
                'description': 'Três métodos ascendentes'
            },
            'THREE_INSIDE_UP': {
                'strength': PatternStrength.STRONG,
                'context_required': 'DOWNTREND',
                'description': 'Três dentro para cima'
            },
            'THREE_OUTSIDE_UP': {
                'strength': PatternStrength.STRONG,
                'context_required': 'DOWNTREND',
                'description': 'Três fora para cima'
            },
            
            # PADRÕES BEARISH FORTES (da imagem 1 e 5)
            'SHOOTING_STAR': {
                'strength': PatternStrength.STRONG,
                'context_required': 'UPTREND',
                'description': 'Estrela cadente em tendência de alta'
            },
            'BEARISH_ENGULFING': {
                'strength': PatternStrength.VERY_STRONG,
                'context_required': 'UPTREND',
                'description': 'Engolfo bearish forte'
            },
            'EVENING_STAR': {
                'strength': PatternStrength.VERY_STRONG,
                'context_required': 'UPTREND',
                'description': 'Estrela da tarde completa'
            },
            'DARK_CLOUD_COVER': {
                'strength': PatternStrength.STRONG,
                'context_required': 'UPTREND',
                'description': 'Cobertura de nuvem escura'
            },
            'BEARISH_HARAMI': {
                'strength': PatternStrength.MEDIUM,
                'context_required': 'UPTREND',
                'description': 'Harami bearish'
            },
            'THREE_BLACK_CROWS': {
                'strength': PatternStrength.VERY_STRONG,
                'context_required': 'ANY',
                'description': 'Três corvos negros'
            },
            'HANGING_MAN': {
                'strength': PatternStrength.MEDIUM,
                'context_required': 'UPTREND',
                'description': 'Homem enforcado'
            },
            'TWEEZER_TOP': {
                'strength': PatternStrength.STRONG,
                'context_required': 'UPTREND',
                'description': 'Tweezer Top'
            },
            'BEARISH_MARUBOZU': {
                'strength': PatternStrength.STRONG,
                'context_required': 'ANY',
                'description': 'Marubozu bearish'
            },
            'GRAVESTONE_DOJI': {
                'strength': PatternStrength.MEDIUM,
                'context_required': 'UPTREND',
                'description': 'Doji lápide'
            },
            'BEARISH_SPINNING_TOP': {
                'strength': PatternStrength.WEAK,
                'context_required': 'UPTREND',
                'description': 'Spinning top bearish'
            },
            'FALLING_THREE_METHODS': {
                'strength': PatternStrength.MEDIUM,
                'context_required': 'DOWNTREND',
                'description': 'Três métodos descendentes'
            },
            'THREE_INSIDE_DOWN': {
                'strength': PatternStrength.STRONG,
                'context_required': 'UPTREND',
                'description': 'Três dentro para baixo'
            },
            'THREE_OUTSIDE_DOWN': {
                'strength': PatternStrength.STRONG,
                'context_required': 'UPTREND',
                'description': 'Três fora para baixo'
            },
            
            # PADRÕES DE INDECISÃO (PERIGOSOS - da imagem 3)
            'DOJI': {
                'strength': PatternStrength.DANGEROUS,
                'context_required': 'ANY',
                'description': 'Doji - indecisão total'
            },
            'SPINNING_TOP': {
                'strength': PatternStrength.DANGEROUS,
                'context_required': 'ANY',
                'description': 'Spinning top - incerteza'
            },
            'LONG_LEGGED_DOJI': {
                'strength': PatternStrength.DANGEROUS,
                'context_required': 'ANY',
                'description': 'Doji pernas longas - indecisão'
            },
            'FOUR_PRICE_DOJI': {
                'strength': PatternStrength.DANGEROUS,
                'context_required': 'ANY',
                'description': 'Doji quatro preços - paralização'
            }
        }
    
    def analyze_all_patterns(self, df: pd.DataFrame, trend_analysis: Dict) -> Dict[str, Any]:
        """Analisa todos os padrões com contexto de tendência"""
        
        if len(df) < 5:
            return {
                'patterns_detected': [],
                'dangerous_patterns': [],
                'pattern_score_call': 0,
                'pattern_score_put': 0,
                'pattern_motivos_call': [],
                'pattern_motivos_put': [],
                'overall_confidence': 0.0
            }
        
        # Detectar todos os padrões
        detected_patterns = []
        
        # Padrões de 1 vela
        single_candle_patterns = self._detect_single_candle_patterns(df)
        detected_patterns.extend(single_candle_patterns)
        
        # Padrões de 2 velas
        two_candle_patterns = self._detect_two_candle_patterns(df)
        detected_patterns.extend(two_candle_patterns)
        
        # Padrões de 3 velas
        three_candle_patterns = self._detect_three_candle_patterns(df)
        detected_patterns.extend(three_candle_patterns)
        
        # ANÁLISE CONTEXTUAL (CRÍTICO - TRADE WITH TREND)
        contextualized_patterns = self._analyze_patterns_with_context(
            detected_patterns, trend_analysis
        )
        
        # IDENTIFICAR PADRÕES PERIGOSOS
        dangerous_patterns = [p['name'] for p in contextualized_patterns 
                            if p['final_strength'] == PatternStrength.DANGEROUS.value]
        
        # CALCULAR SCORES FINAIS
        pattern_score_call = sum(p['score_call'] for p in contextualized_patterns)
        pattern_score_put = sum(p['score_put'] for p in contextualized_patterns)
        
        # GERAR MOTIVOS (apenas padrões fortes)
        motivos_call = [p['description'] for p in contextualized_patterns 
                       if p['score_call'] > 20]
        motivos_put = [p['description'] for p in contextualized_patterns 
                      if p['score_put'] > 20]
        
        # CONFIANÇA GERAL
        overall_confidence = self._calculate_overall_confidence(contextualized_patterns)
        
        return {
            'patterns_detected': [p['name'] for p in contextualized_patterns],
            'dangerous_patterns': dangerous_patterns,
            'pattern_score_call': min(pattern_score_call, 100),  # Limitar score
            'pattern_score_put': min(pattern_score_put, 100),
            'pattern_motivos_call': motivos_call[:3],  # Máximo 3 motivos
            'pattern_motivos_put': motivos_put[:3],
            'overall_confidence': overall_confidence,
            'detailed_patterns': contextualized_patterns
        }
    
    def analyze_dangerous_patterns(self, df: pd.DataFrame, analise_completa: Dict) -> Dict[str, Any]:
        """🚫 BLOQUEIA PADRÕES PERIGOSOS que devem impedir trades"""
        
        if len(df) < 3:
            return {'should_block': False, 'block_reason': '', 'dangerous_patterns': []}
        
        dangerous_found = []
        block_reasons = []
        
        # DETECTAR DOJI E SPINNING TOPS
        current = df.iloc[-1]
        o, h, l, c = current['open'], current['high'], current['low'], current['close']
        
        body = abs(c - o)
        total_range = h - l
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        
        # DOJI (indecisão total)
        if total_range > 0 and body <= total_range * 0.1:
            dangerous_found.append('DOJI')
            block_reasons.append('Doji detectado - indecisão total')
        
        # SPINNING TOP (incerteza)
        elif (total_range > 0 and body <= total_range * 0.3 and 
              upper_shadow >= body * 0.5 and lower_shadow >= body * 0.5):
            dangerous_found.append('SPINNING_TOP')
            block_reasons.append('Spinning Top - incerteza no mercado')
        
        # PADRÕES CONFLITANTES
        motivos_call = analise_completa.get('motivos_call', [])
        motivos_put = analise_completa.get('motivos_put', [])
        
        # Se há muitos sinais conflitantes
        if len(motivos_call) > 0 and len(motivos_put) > 0:
            score_call = analise_completa.get('score_call', 0)
            score_put = analise_completa.get('score_put', 0)
            
            # Sinais muito equilibrados = perigoso
            if abs(score_call - score_put) < 20 and max(score_call, score_put) < 200:
                dangerous_found.append('CONFLICTING_SIGNALS')
                block_reasons.append('Sinais conflitantes - sem direção clara')
        
        # VOLATILIDADE MUITO BAIXA
        volatilidade = analise_completa.get('volatilidade', 1.0)
        if volatilidade < 0.1:
            dangerous_found.append('LOW_VOLATILITY')
            block_reasons.append('Volatilidade extremamente baixa')
        
        # DECISÃO DE BLOQUEIO
        should_block = len(dangerous_found) >= 1
        
        return {
            'should_block': should_block,
            'block_reason': ', '.join(block_reasons[:2]),
            'dangerous_patterns': dangerous_found
        }
    
    def _detect_single_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta padrões de vela única baseados nas imagens"""
        
        if len(df) < 1:
            return []
        
        patterns = []
        current = df.iloc[-1]
        o, h, l, c = current['open'], current['high'], current['low'], current['close']
        
        body = abs(c - o)
        total_range = h - l
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        
        if total_range == 0:
            return patterns
        
        # HAMMER (Imagem 1 e 4)
        if (lower_shadow >= body * 2 and upper_shadow <= body * 0.3 and 
            body >= total_range * 0.1 and c > o):
            patterns.append({
                'name': 'HAMMER',
                'type': 'BULLISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 1
            })
        
        # SHOOTING STAR (Imagem 1 e 4)
        if (upper_shadow >= body * 2 and lower_shadow <= body * 0.3 and 
            body >= total_range * 0.1 and c < o):
            patterns.append({
                'name': 'SHOOTING_STAR',
                'type': 'BEARISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 1
            })
        
        # DOJI (Imagem 3 - PERIGOSO)
        if body <= total_range * 0.1:
            patterns.append({
                'name': 'DOJI',
                'type': 'INDECISION',
                'strength': PatternStrength.DANGEROUS.value,
                'candles': 1
            })
        
        # SPINNING TOP (Imagem 3 - PERIGOSO)
        elif (body <= total_range * 0.3 and 
              upper_shadow >= body * 0.5 and lower_shadow >= body * 0.5):
            patterns.append({
                'name': 'SPINNING_TOP',
                'type': 'INDECISION',
                'strength': PatternStrength.DANGEROUS.value,
                'candles': 1
            })
        
        # MARUBOZU BULLISH (Imagem 1)
        elif (c > o and upper_shadow <= total_range * 0.05 and 
              lower_shadow <= total_range * 0.05 and body >= total_range * 0.9):
            patterns.append({
                'name': 'BULLISH_MARUBOZU',
                'type': 'BULLISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 1
            })
        
        # MARUBOZU BEARISH (Imagem 3)
        elif (c < o and upper_shadow <= total_range * 0.05 and 
              lower_shadow <= total_range * 0.05 and body >= total_range * 0.9):
            patterns.append({
                'name': 'BEARISH_MARUBOZU',
                'type': 'BEARISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 1
            })
        
        # DRAGONFLY DOJI (Imagem 1)
        elif (body <= total_range * 0.1 and upper_shadow <= total_range * 0.1 and
              lower_shadow >= total_range * 0.6):
            patterns.append({
                'name': 'DRAGONFLY_DOJI',
                'type': 'BULLISH',
                'strength': PatternStrength.MEDIUM.value,
                'candles': 1
            })
        
        # GRAVESTONE DOJI (Imagem 3)
        elif (body <= total_range * 0.1 and lower_shadow <= total_range * 0.1 and
              upper_shadow >= total_range * 0.6):
            patterns.append({
                'name': 'GRAVESTONE_DOJI',
                'type': 'BEARISH',
                'strength': PatternStrength.MEDIUM.value,
                'candles': 1
            })
        
        # INVERTED HAMMER (Imagem 1)
        elif (upper_shadow >= body * 2 and lower_shadow <= body * 0.3 and 
              body >= total_range * 0.1 and c > o):
            patterns.append({
                'name': 'INVERTED_HAMMER',
                'type': 'BULLISH',
                'strength': PatternStrength.MEDIUM.value,
                'candles': 1
            })
        
        # HANGING MAN (Similar ao Hammer mas em uptrend)
        elif (lower_shadow >= body * 2 and upper_shadow <= body * 0.3 and 
              body >= total_range * 0.1 and c < o):
            patterns.append({
                'name': 'HANGING_MAN',
                'type': 'BEARISH',
                'strength': PatternStrength.MEDIUM.value,
                'candles': 1
            })
        
        return patterns
    
    def _detect_two_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta padrões de duas velas baseados nas imagens"""
        
        if len(df) < 2:
            return []
        
        patterns = []
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # BULLISH ENGULFING (Imagem 1 e 5)
        if (previous['close'] < previous['open'] and  # Vela anterior bearish
            current['close'] > current['open'] and   # Vela atual bullish
            current['open'] < previous['close'] and  # Abre abaixo do fechamento anterior
            current['close'] > previous['open']):    # Fecha acima da abertura anterior
            
            engulf_strength = abs(current['close'] - current['open']) / abs(previous['close'] - previous['open'])
            
            if engulf_strength > 1.2:  # Engolfamento significativo
                patterns.append({
                    'name': 'BULLISH_ENGULFING',
                    'type': 'BULLISH',
                    'strength': PatternStrength.VERY_STRONG.value,
                    'candles': 2
                })
        
        # BEARISH ENGULFING (Imagem 5)
        if (previous['close'] > previous['open'] and  # Vela anterior bullish
            current['close'] < current['open'] and   # Vela atual bearish
            current['open'] > previous['close'] and  # Abre acima do fechamento anterior
            current['close'] < previous['open']):    # Fecha abaixo da abertura anterior
            
            engulf_strength = abs(current['close'] - current['open']) / abs(previous['close'] - previous['open'])
            
            if engulf_strength > 1.2:  # Engolfamento significativo
                patterns.append({
                    'name': 'BEARISH_ENGULFING',
                    'type': 'BEARISH',
                    'strength': PatternStrength.VERY_STRONG.value,
                    'candles': 2
                })
        
        # HARAMI PATTERNS (Imagem 1)
        current_body = abs(current['close'] - current['open'])
        previous_body = abs(previous['close'] - previous['open'])
        
        if (current_body < previous_body * 0.7 and
            current['high'] < previous['high'] and current['low'] > previous['low']):
            
            if previous['close'] < previous['open']:  # Previous bearish
                patterns.append({
                    'name': 'BULLISH_HARAMI',
                    'type': 'BULLISH',
                    'strength': PatternStrength.MEDIUM.value,
                    'candles': 2
                })
            else:  # Previous bullish
                patterns.append({
                    'name': 'BEARISH_HARAMI',
                    'type': 'BEARISH',
                    'strength': PatternStrength.MEDIUM.value,
                    'candles': 2
                })
        
        # PIERCING PATTERN (Imagem 1)
        if (previous['close'] < previous['open'] and  # Vela anterior bearish
            current['close'] > current['open'] and   # Vela atual bullish
            current['open'] < previous['low'] and    # Abre abaixo da mínima anterior
            current['close'] > (previous['open'] + previous['close']) / 2):  # Fecha acima do meio
            
            patterns.append({
                'name': 'PIERCING_PATTERN',
                'type': 'BULLISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 2
            })
        
        # DARK CLOUD COVER (Imagem 5)
        if (previous['close'] > previous['open'] and  # Vela anterior bullish
            current['close'] < current['open'] and   # Vela atual bearish
            current['open'] > previous['high'] and   # Abre acima da máxima anterior
            current['close'] < (previous['open'] + previous['close']) / 2):  # Fecha abaixo do meio
            
            patterns.append({
                'name': 'DARK_CLOUD_COVER',
                'type': 'BEARISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 2
            })
        
        # TWEEZER BOTTOM (Imagem 1 e 5)
        if (abs(previous['low'] - current['low']) / previous['low'] < 0.002 and  # Mínimas iguais
            previous['close'] < previous['open'] and  # Primeira bearish
            current['close'] > current['open']):     # Segunda bullish
            
            patterns.append({
                'name': 'TWEEZER_BOTTOM',
                'type': 'BULLISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 2
            })
        
        # TWEEZER TOP (Imagem 5)
        if (abs(previous['high'] - current['high']) / previous['high'] < 0.002 and  # Máximas iguais
            previous['close'] > previous['open'] and  # Primeira bullish
            current['close'] < current['open']):     # Segunda bearish
            
            patterns.append({
                'name': 'TWEEZER_TOP',
                'type': 'BEARISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 2
            })
        
        return patterns
    
    def _detect_three_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta padrões de três velas baseados nas imagens"""
        
        if len(df) < 3:
            return []
        
        patterns = []
        candle1 = df.iloc[-3]
        candle2 = df.iloc[-2] 
        candle3 = df.iloc[-1]
        
        # MORNING STAR (Imagem 1 e 5)
        if (candle1['close'] < candle1['open'] and  # Primeira vela bearish
            candle3['close'] > candle3['open']):    # Terceira vela bullish
            
            body2 = abs(candle2['close'] - candle2['open'])
            body1 = abs(candle1['close'] - candle1['open'])
            body3 = abs(candle3['close'] - candle3['open'])
            
            if (body2 <= min(body1, body3) * 0.5 and  # Segunda vela pequena
                candle3['close'] > (candle1['open'] + candle1['close']) / 2):  # Terceira penetra na primeira
                
                patterns.append({
                    'name': 'MORNING_STAR',
                    'type': 'BULLISH',
                    'strength': PatternStrength.VERY_STRONG.value,
                    'candles': 3
                })
        
        # EVENING STAR (Imagem 5)
        if (candle1['close'] > candle1['open'] and  # Primeira vela bullish
            candle3['close'] < candle3['open']):    # Terceira vela bearish
            
            body2 = abs(candle2['close'] - candle2['open'])
            body1 = abs(candle1['close'] - candle1['open'])
            body3 = abs(candle3['close'] - candle3['open'])
            
            if (body2 <= min(body1, body3) * 0.5 and  # Segunda vela pequena
                candle3['close'] < (candle1['open'] + candle1['close']) / 2):  # Terceira penetra na primeira
                
                patterns.append({
                    'name': 'EVENING_STAR',
                    'type': 'BEARISH',
                    'strength': PatternStrength.VERY_STRONG.value,
                    'candles': 3
                })
        
        # THREE WHITE SOLDIERS (Imagem 1 e 5)
        if (candle1['close'] > candle1['open'] and
            candle2['close'] > candle2['open'] and 
            candle3['close'] > candle3['open'] and
            candle2['close'] > candle1['close'] and
            candle3['close'] > candle2['close'] and
            candle2['open'] > candle1['open'] and
            candle3['open'] > candle2['open']):
            
            patterns.append({
                'name': 'THREE_WHITE_SOLDIERS',
                'type': 'BULLISH',
                'strength': PatternStrength.VERY_STRONG.value,
                'candles': 3
            })
        
        # THREE BLACK CROWS (Imagem 5)
        if (candle1['close'] < candle1['open'] and
            candle2['close'] < candle2['open'] and 
            candle3['close'] < candle3['open'] and
            candle2['close'] < candle1['close'] and
            candle3['close'] < candle2['close'] and
            candle2['open'] < candle1['open'] and
            candle3['open'] < candle2['open']):
            
            patterns.append({
                'name': 'THREE_BLACK_CROWS',
                'type': 'BEARISH',
                'strength': PatternStrength.VERY_STRONG.value,
                'candles': 3
            })
        
        # THREE INSIDE UP (Imagem 1)
        if (candle1['close'] < candle1['open'] and  # Primeira bearish
            candle2['close'] > candle2['open'] and  # Segunda bullish pequena
            candle3['close'] > candle3['open'] and  # Terceira bullish
            candle2['high'] < candle1['high'] and candle2['low'] > candle1['low'] and  # Harami
            candle3['close'] > candle1['close']):   # Confirmação
            
            patterns.append({
                'name': 'THREE_INSIDE_UP',
                'type': 'BULLISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 3
            })
        
        # THREE INSIDE DOWN (Imagem 5)
        if (candle1['close'] > candle1['open'] and  # Primeira bullish
            candle2['close'] < candle2['open'] and  # Segunda bearish pequena
            candle3['close'] < candle3['open'] and  # Terceira bearish
            candle2['high'] < candle1['high'] and candle2['low'] > candle1['low'] and  # Harami
            candle3['close'] < candle1['close']):   # Confirmação
            
            patterns.append({
                'name': 'THREE_INSIDE_DOWN',
                'type': 'BEARISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 3
            })
        
        # THREE OUTSIDE UP (Imagem 1)
        if (candle1['close'] < candle1['open'] and  # Primeira bearish
            candle2['close'] > candle2['open'] and  # Segunda bullish engolfando
            candle3['close'] > candle3['open'] and  # Terceira bullish
            candle2['open'] < candle1['close'] and candle2['close'] > candle1['open'] and  # Engolfamento
            candle3['close'] > candle2['close']):   # Confirmação
            
            patterns.append({
                'name': 'THREE_OUTSIDE_UP',
                'type': 'BULLISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 3
            })
        
        # THREE OUTSIDE DOWN (Imagem 5)
        if (candle1['close'] > candle1['open'] and  # Primeira bullish
            candle2['close'] < candle2['open'] and  # Segunda bearish engolfando
            candle3['close'] < candle3['open'] and  # Terceira bearish
            candle2['open'] > candle1['close'] and candle2['close'] < candle1['open'] and  # Engolfamento
            candle3['close'] < candle2['close']):   # Confirmação
            
            patterns.append({
                'name': 'THREE_OUTSIDE_DOWN',
                'type': 'BEARISH',
                'strength': PatternStrength.STRONG.value,
                'candles': 3
            })
        
        return patterns
    
    def _analyze_patterns_with_context(self, patterns: List[Dict], trend_analysis: Dict) -> List[Dict]:
        """🔥 CRÍTICO: Analisa padrões com contexto de tendência (TRADE WITH TREND)"""
        
        contextualized = []
        macro_trend = trend_analysis.get('macro_trend', 'NEUTRAL')
        micro_trend = trend_analysis.get('micro_trend', 'NEUTRAL')
        
        for pattern in patterns:
            pattern_name = pattern['name']
            pattern_type = pattern['type']
            base_strength = pattern['strength']
            
            # ANÁLISE CONTEXTUAL CRÍTICA (TRADE WITH TREND)
            context_multiplier = 1.0
            final_score_call = 0
            final_score_put = 0
            
            if pattern_type == 'BULLISH':
                # Padrão bullish NA DIREÇÃO da tendência = FORTE
                if 'BULLISH' in macro_trend:
                    context_multiplier = 1.5
                elif 'BEARISH' in macro_trend:
                    context_multiplier = 0.3  # CONTRA tendência = FRACO
                
                # Aplicar score final
                final_score_call = base_strength * context_multiplier
                
            elif pattern_type == 'BEARISH':
                # Padrão bearish NA DIREÇÃO da tendência = FORTE
                if 'BEARISH' in macro_trend:
                    context_multiplier = 1.5
                elif 'BULLISH' in macro_trend:
                    context_multiplier = 0.3  # CONTRA tendência = FRACO
                
                # Aplicar score final
                final_score_put = base_strength * context_multiplier
                
            elif pattern_type == 'INDECISION':
                # Padrões de indecisão sempre perigosos
                context_multiplier = 1.0
                final_score_call = PatternStrength.DANGEROUS.value
                final_score_put = PatternStrength.DANGEROUS.value
            
            # CONFLUÊNCIA MACRO/MICRO
            confluence_bonus = 0
            if trend_analysis.get('macro_micro_confluence', False):
                confluence_direction = trend_analysis.get('confluence_direction')
                if ((pattern_type == 'BULLISH' and confluence_direction == 'BULLISH') or
                    (pattern_type == 'BEARISH' and confluence_direction == 'BEARISH')):
                    confluence_bonus = 15
                    final_score_call += confluence_bonus if pattern_type == 'BULLISH' else 0
                    final_score_put += confluence_bonus if pattern_type == 'BEARISH' else 0
            
            # DESCRIÇÃO COM CONTEXTO
            if pattern_name in self.pattern_database:
                base_description = self.pattern_database[pattern_name]['description']
                required_context = self.pattern_database[pattern_name]['context_required']
                
                # Verificar se contexto está correto
                context_ok = True
                if required_context == 'UPTREND' and 'BEARISH' in macro_trend:
                    context_ok = False
                elif required_context == 'DOWNTREND' and 'BULLISH' in macro_trend:
                    context_ok = False
                
                description = f"{base_description}"
                if context_multiplier > 1.0:
                    description += " (Contexto Favorável)"
                elif context_multiplier < 1.0:
                    description += " (Contra Tendência)"
                
                if confluence_bonus > 0:
                    description += " + Confluência"
                
            else:
                description = f"Padrão {pattern_name}"
            
            contextualized.append({
                'name': pattern_name,
                'type': pattern_type,
                'base_strength': base_strength,
                'context_multiplier': context_multiplier,
                'final_strength': base_strength * context_multiplier,
                'score_call': max(0, final_score_call),
                'score_put': max(0, final_score_put),
                'description': description,
                'confluence_bonus': confluence_bonus,
                'context_ok': context_ok if 'context_ok' in locals() else True
            })
        
        return contextualized
    
    def _calculate_overall_confidence(self, patterns: List[Dict]) -> float:
        """Calcula confiança geral baseada nos padrões detectados"""
        
        if not patterns:
            return 0.0
        
        # Padrões perigosos reduzem confiança drasticamente
        dangerous_count = sum(1 for p in patterns if p['final_strength'] < 0)
        if dangerous_count > 0:
            return 0.2  # Baixa confiança
        
        # Calcular confiança baseada na força dos padrões
        strong_patterns = [p for p in patterns if p['final_strength'] >= 40]
        medium_patterns = [p for p in patterns if 20 <= p['final_strength'] < 40]
        weak_patterns = [p for p in patterns if 0 < p['final_strength'] < 20]
        
        confidence = 0.5  # Base
        
        # Padrões fortes aumentam confiança
        confidence += len(strong_patterns) * 0.2
        confidence += len(medium_patterns) * 0.1
        confidence -= len(weak_patterns) * 0.05
        
        # Confluência aumenta confiança
        confluence_patterns = [p for p in patterns if p.get('confluence_bonus', 0) > 0]
        confidence += len(confluence_patterns) * 0.15
        
        return min(1.0, max(0.0, confidence))

class PriceActionSupreme1000Velas:
    """🚀 ANÁLISE PRICE ACTION HISTÓRICA DE 1000 VELAS"""
    
    @staticmethod
    def analyze_historical_price_action(df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa price action das últimas 1000 velas (ou máximo disponível)"""
        
        # Usar máximo de velas disponíveis, até 1000
        max_velas = min(len(df), 1000)
        if max_velas < 100:
            return {
                'historical_sr': [],
                'major_trends': [],
                'key_levels': [],
                'macro_context': 'INSUFFICIENT_DATA'
            }
        
        historical_data = df.tail(max_velas)
        
        # DETECTAR SUPORTES/RESISTÊNCIAS HISTÓRICOS
        historical_sr = PriceActionSupreme1000Velas._detect_historical_sr(historical_data)
        
        # DETECTAR TENDÊNCIAS PRINCIPAIS
        major_trends = PriceActionSupreme1000Velas._detect_major_trends(historical_data)
        
        # IDENTIFICAR NÍVEIS CHAVE
        key_levels = PriceActionSupreme1000Velas._identify_key_levels(historical_data)
        
        # CONTEXTO MACRO
        macro_context = PriceActionSupreme1000Velas._determine_macro_context(historical_data)
        
        return {
            'historical_sr': historical_sr,
            'major_trends': major_trends,
            'key_levels': key_levels,
            'macro_context': macro_context,
            'velas_analisadas': max_velas,
            'timeframe_coverage': f'{max_velas} velas'
        }
    
    @staticmethod
    def _detect_historical_sr(df: pd.DataFrame) -> List[Dict]:
        """Detecta suportes e resistências históricos"""
        
        highs = df['high'].values
        lows = df['low'].values
        current_price = df['close'].iloc[-1]
        
        # Detectar níveis históricos importantes
        historical_levels = []
        
        # Resistências históricas
        for i in range(10, len(highs) - 10):
            if all(highs[i] >= highs[j] for j in range(i-10, i+10) if j != i):
                touches = sum(1 for h in highs if abs(h - highs[i]) / highs[i] < 0.005)
                if touches >= 3:
                    historical_levels.append({
                        'level': highs[i],
                        'type': 'RESISTANCE',
                        'strength': touches,
                        'distance_pct': abs(current_price - highs[i]) / current_price * 100,
                        'age': len(highs) - i
                    })
        
        # Suportes históricos
        for i in range(10, len(lows) - 10):
            if all(lows[i] <= lows[j] for j in range(i-10, i+10) if j != i):
                touches = sum(1 for l in lows if abs(l - lows[i]) / lows[i] < 0.005)
                if touches >= 3:
                    historical_levels.append({
                        'level': lows[i],
                        'type': 'SUPPORT',
                        'strength': touches,
                        'distance_pct': abs(current_price - lows[i]) / current_price * 100,
                        'age': len(lows) - i
                    })
        
        # Ordenar por relevância (força + proximidade)
        for level in historical_levels:
            level['relevance'] = level['strength'] / (1 + level['distance_pct']/10)
        
        return sorted(historical_levels, key=lambda x: x['relevance'], reverse=True)[:10]
    
    @staticmethod
    def _detect_major_trends(df: pd.DataFrame) -> List[Dict]:
        """Detecta tendências principais no histórico"""
        
        closes = df['close'].values
        periods = [50, 100, 200, 500]  # Diferentes períodos para análise
        
        trends = []
        
        for period in periods:
            if len(closes) >= period:
                recent_closes = closes[-period:]
                
                # Calcular tendência linear
                x = np.arange(len(recent_closes))
                slope = np.polyfit(x, recent_closes, 1)[0]
                
                # Determinar direção e força
                slope_pct = (slope * len(recent_closes)) / recent_closes[0] * 100
                
                if slope_pct > 5:
                    direction = 'STRONG_BULLISH'
                elif slope_pct > 1:
                    direction = 'BULLISH'
                elif slope_pct < -5:
                    direction = 'STRONG_BEARISH'
                elif slope_pct < -1:
                    direction = 'BEARISH'
                else:
                    direction = 'SIDEWAYS'
                
                trends.append({
                    'period': period,
                    'direction': direction,
                    'slope_pct': slope_pct,
                    'strength': abs(slope_pct),
                    'start_price': recent_closes[0],
                    'end_price': recent_closes[-1]
                })
        
        return trends
    
    @staticmethod
    def _identify_key_levels(df: pd.DataFrame) -> List[Dict]:
        """Identifica níveis chave psicológicos e técnicos"""
        
        current_price = df['close'].iloc[-1]
        
        # Níveis psicológicos (números redondos)
        psychological_levels = []
        
        # Encontrar números redondos próximos
        price_magnitude = 10 ** (len(str(int(current_price))) - 2)
        
        for i in range(-5, 6):
            level = round(current_price / price_magnitude) * price_magnitude + (i * price_magnitude)
            if level > 0:
                distance_pct = abs(current_price - level) / current_price * 100
                if distance_pct <= 10:  # Dentro de 10%
                    psychological_levels.append({
                        'level': level,
                        'type': 'PSYCHOLOGICAL',
                        'distance_pct': distance_pct,
                        'description': f'Nível {level:.0f}'
                    })
        
        # Níveis de Fibonacci
        highs = df['high'].values
        lows = df['low'].values
        
        # Usar swing high/low mais recentes significativos
        recent_high = np.max(highs[-100:])
        recent_low = np.min(lows[-100:])
        
        fib_levels = []
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            fib_level = recent_low + (recent_high - recent_low) * ratio
            distance_pct = abs(current_price - fib_level) / current_price * 100
            
            if distance_pct <= 5:  # Próximo de nível Fib
                fib_levels.append({
                    'level': fib_level,
                    'type': 'FIBONACCI',
                    'ratio': ratio,
                    'distance_pct': distance_pct,
                    'description': f'Fib {ratio:.1%}'
                })
        
        return psychological_levels + fib_levels
    
    @staticmethod
    def _determine_macro_context(df: pd.DataFrame) -> str:
        """Determina contexto macro baseado em análise histórica"""
        
        closes = df['close'].values
        
        if len(closes) < 200:
            return 'INSUFFICIENT_DATA'
        
        # Médias de longo prazo
        ma50 = np.mean(closes[-50:])
        ma200 = np.mean(closes[-200:])
        current = closes[-1]
        
        # Volatilidade histórica
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * 100
        
        # Determinar contexto
        if current > ma50 > ma200 and volatility < 2:
            return 'STABLE_UPTREND'
        elif current > ma50 > ma200 and volatility > 3:
            return 'VOLATILE_UPTREND'
        elif current < ma50 < ma200 and volatility < 2:
            return 'STABLE_DOWNTREND'
        elif current < ma50 < ma200 and volatility > 3:
            return 'VOLATILE_DOWNTREND'
        elif abs(ma50 - ma200) / ma200 < 0.02:
            return 'CONSOLIDATION'
        else:
            return 'TRANSITIONAL'

print("✅ PATTERN RECOGNITION SUPREME - SISTEMA AVANÇADO DE PADRÕES + TENDÊNCIAS + 1000 VELAS CARREGADO!")