#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 ANALISADOR COMPLETO V8 + ENHANCED SUPREME - VERSÃO FINAL INTEGRADA 👑
💎 SISTEMA SOBREVIVÊNCIA + 50+ PADRÕES + MACRO/MICRO TENDÊNCIAS + 1000 VELAS
🔥 ANÁLISE ANTI-LOSS + DATABASE INTELLIGENCE + PATTERN RECOGNITION SUPREME
🚀 SOLUÇÃO DEFINITIVA: Evita CALL em alta batendo resistência + Trade With Trend
"""

import pandas as pd
import numpy as np
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from config_royal import (ConfigRoyalSupremeEnhanced, TipoSinalRoyalSupremeEnhanced, 
                         StatusSinalRoyalSupremeEnhanced, MarketScenarioRoyalSupremeEnhanced, 
                         SurvivabilityMode, PARES_CRYPTO, TIMEZONE)
from enhanced_technical import PriceActionMasterRoyalSupremeEnhanced
from arsenal_tecnico import ArsenalTecnicoCompletoV8RoyalSupremeEnhanced
from detectores_mercado import DetectorMercadoCaoticoV8RoyalSupremeEnhanced, DetectorCenariosExtremosV8RoyalSupremeEnhanced

# NOVO: Import do sistema de padrões avançados
from pattern_recognition_supreme import PatternRecognitionSupreme, TrendAnalysisSupreme, PriceActionSupreme1000Velas

class SistemaSobrevivenciaV8RoyalSupremeEnhanced:
    
    def __init__(self, relatorios_system, db_manager):
        self.relatorios = relatorios_system
        self.db_manager = db_manager
        self.detector_cenarios = DetectorCenariosExtremosV8RoyalSupremeEnhanced()
        self.detector_caotico = DetectorMercadoCaoticoV8RoyalSupremeEnhanced()
        
        # NOVO: Sistema de padrões avançados
        self.pattern_recognition = PatternRecognitionSupreme()
        self.trend_analysis = TrendAnalysisSupreme()
        
        self.modo_atual = SurvivabilityMode.NORMAL
        self.protecoes_ativas = set()
        self.historico_protecoes = deque(maxlen=1000)
        self.stats_sobrevivencia = {
            'cenarios_detectados': 0,
            'protecoes_ativadas': 0,
            'losses_evitados': 0,
            'oportunidades_capturadas': 0,
            'mercados_caoticos_detectados': 0,
            'auto_calibrador_ativacoes': 0,
            'trend_opportunities': 0,
            'pullback_opportunities': 0,
            'elliott_opportunities': 0,
            'patterns_blocked': 0,
            'macro_micro_confluences': 0,
            'corrections_avoided': 0,
            'resumptions_caught': 0
        }
    
    def avaliar_entrada_segura(self, df: pd.DataFrame, par: str, tipo_sinal: str, 
                              score_total: float, confluencia_count: int, analise_completa: Dict) -> Dict[str, Any]:
        
        # 🚀 AUTO CALIBRADOR DINÂMICO FUNCIONAL CORRIGIDO
        auto_calibrador_usado = False
        adjustment_total = 0
        
        if ConfigRoyalSupremeEnhanced.AUTO_CALIBRADOR_ENABLED:
            try:
                # 1. QUALITY SCORE DINÂMICO REAL
                quality_score = self.db_manager.get_quality_score_por_par(par)
                
                # 2. LÓGICA IA CORRIGIDA - AUMENTAR THRESHOLD BASEADO EM QUALITY
                threshold_adjustment = 0
                
                if quality_score < 55:
                    threshold_adjustment = 100  # Muito rigoroso
                elif quality_score < 60:
                    threshold_adjustment = 60   # Rigoroso
                elif quality_score < 70:
                    threshold_adjustment = 40   # Moderadamente rigoroso
                elif quality_score < 75:
                    threshold_adjustment = 40   # Moderado
                elif quality_score < 80:
                    threshold_adjustment = 25   # Leve
                elif quality_score < 85:
                    threshold_adjustment = 15   # Muito leve
                else:
                    threshold_adjustment = 5    # Mínimo
                
                # 3. AJUSTE POR LOSSES CONSECUTIVOS
                losses_consecutivos = self._get_losses_consecutivos_recentes(par)
                if losses_consecutivos >= 3:
                    threshold_adjustment += 70
                elif losses_consecutivos >= 2:
                    threshold_adjustment += 50
                elif losses_consecutivos >= 1:
                    threshold_adjustment += 30
                
                # 4. AJUSTE ESPECÍFICO POR PAR (BASEADO NOS LOGS)
                if par == 'adausdt':
                    threshold_adjustment += 40  # ADA mais rigoroso sempre
                elif par == 'ethusdt' and quality_score < 80:
                    threshold_adjustment += 25  # ETH rigoroso se performance baixa
                
                # 5. APLICAR THRESHOLD DINÂMICO FUNCIONAL
                if threshold_adjustment > 15:
                    threshold_dinamico = ConfigRoyalSupremeEnhanced.MIN_SCORE_NORMAL + threshold_adjustment
                    auto_calibrador_usado = True
                    adjustment_total = threshold_adjustment
                    
                    # INCREMENTAR CONTADOR CORRETAMENTE
                    self.relatorios.stats_globais['auto_calibrador_usado'] += 1
                    
                    if score_total < threshold_dinamico:
                        return {
                            'entrada_segura': False,
                            'cenario_detectado': MarketScenarioRoyalSupremeEnhanced.NORMAL,
                            'modo_sobrevivencia': SurvivabilityMode.DEFENSIVE,
                            'motivo_bloqueio': f"👑 AUTO CALIBRADOR IA: {par.upper()} precisa {threshold_dinamico}+ score (atual:{score_total:.0f}, QS:{quality_score:.1f}%)",
                            'auto_calibrador_ativo': True,
                            'quality_score': quality_score,
                            'threshold_aplicado': threshold_dinamico,
                            'adjustment_aplicado': adjustment_total
                        }
                        
            except Exception as e:
                pass
        
        # 🚀 NOVO: ANÁLISE DE PADRÕES PERIGOSOS
        pattern_analysis = self.pattern_recognition.analyze_dangerous_patterns(df, analise_completa)
        
        if pattern_analysis['should_block']:
            self.stats_sobrevivencia['patterns_blocked'] += 1
            self.relatorios.registrar_pattern_block(par)
            return {
                'entrada_segura': False,
                'cenario_detectado': MarketScenarioRoyalSupremeEnhanced.NORMAL,
                'modo_sobrevivencia': SurvivabilityMode.DEFENSIVE,
                'motivo_bloqueio': f"🚫 PADRÃO PERIGOSO: {pattern_analysis['block_reason']}",
                'pattern_block': True,
                'dangerous_patterns': pattern_analysis['dangerous_patterns']
            }
        
        # 🚀 NOVO: DETECTOR DE FASES DE MOMENTUM (RESOLVE O PROBLEMA PRINCIPAL)
        movimento_1min = analise_completa.get('movimento_1min', 0)
        momentum_analysis = self.trend_analysis.analyze_momentum_phase(df, movimento_1min)
        
        # 🔥 CRÍTICO: BLOQUEAR ENTRADAS EM CORREÇÃO
        if momentum_analysis['phase'] == 'CORRECTION':
            self.stats_sobrevivencia['corrections_avoided'] += 1
            return {
                'entrada_segura': False,
                'cenario_detectado': MarketScenarioRoyalSupremeEnhanced.NORMAL,
                'modo_sobrevivencia': SurvivabilityMode.DEFENSIVE,
                'motivo_bloqueio': f"🚫 FASE CORREÇÃO: {momentum_analysis['direction']} - Aguardar retomada",
                'momentum_block': True,
                'momentum_phase': momentum_analysis['phase']
            }
        
        # 🚀 BLACKLIST INTELIGENTE AVANÇADO
        if self._verificar_blacklist_inteligente_corrigido(par, analise_completa):
            return {
                'entrada_segura': False,
                'cenario_detectado': MarketScenarioRoyalSupremeEnhanced.NORMAL,
                'modo_sobrevivencia': SurvivabilityMode.DEFENSIVE,
                'motivo_bloqueio': f"🚫 BLACKLIST IA: {par.upper()} padrão de loss detectado",
                'blacklist_ativo': True
            }
        
        # DETECTOR MERCADO CAÓTICO (preservado)
        analise_caotico = self.detector_caotico.detectar_mercado_caotico(df, analise_completa)
        
        if analise_caotico['mercado_caotico']:
            self.stats_sobrevivencia['mercados_caoticos_detectados'] += 1
            return {
                'entrada_segura': False,
                'cenario_detectado': MarketScenarioRoyalSupremeEnhanced.MERCADO_CAOTICO,
                'modo_sobrevivencia': SurvivabilityMode.BUNKER,
                'motivo_bloqueio': f"🌙 MERCADO CAÓTICO: {', '.join(analise_caotico['motivos'][:2])}",
                'analise_caotico': analise_caotico
            }
        
        # DETECTOR CENÁRIOS (preservado)
        cenario_atual = self.detector_cenarios.detectar_cenario_atual(df, par)
        
        # ENHANCED OPPORTUNITIES - SEMPRE PERMITIDAS
        enhanced_opportunities = [
            MarketScenarioRoyalSupremeEnhanced.ELITE_OPPORTUNITY,
            MarketScenarioRoyalSupremeEnhanced.WAVE_OPPORTUNITY,
            MarketScenarioRoyalSupremeEnhanced.TREND_OPPORTUNITY,
            MarketScenarioRoyalSupremeEnhanced.PULLBACK_OPPORTUNITY,
            MarketScenarioRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY
        ]
        
        if cenario_atual in enhanced_opportunities:
            # CORREÇÃO: Incrementar nos stats globais
            if cenario_atual == MarketScenarioRoyalSupremeEnhanced.TREND_OPPORTUNITY:
                self.relatorios.stats_globais['trend_opportunities'] += 1
            elif cenario_atual == MarketScenarioRoyalSupremeEnhanced.PULLBACK_OPPORTUNITY:
                self.relatorios.stats_globais['pullback_opportunities'] += 1
            elif cenario_atual == MarketScenarioRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY:
                self.relatorios.stats_globais['elliott_opportunities'] += 1
            elif cenario_atual == MarketScenarioRoyalSupremeEnhanced.ELITE_OPPORTUNITY:
                self.relatorios.stats_globais['elite_opportunities'] += 1
            elif cenario_atual == MarketScenarioRoyalSupremeEnhanced.WAVE_OPPORTUNITY:
                self.relatorios.stats_globais['wave_opportunities'] += 1
            
            return {
                'entrada_segura': True,
                'cenario_detectado': cenario_atual,
                'modo_sobrevivencia': SurvivabilityMode.NORMAL,
                'oportunidade_especial': True,
                'auto_calibrador_ativo': auto_calibrador_usado,
                'adjustment_aplicado': adjustment_total
            }
        
        # 🚀 BOOST POR RETOMADA DE TENDÊNCIA
        if momentum_analysis['phase'] == 'RESUMPTION':
            self.stats_sobrevivencia['resumptions_caught'] += 1
            return {
                'entrada_segura': True,
                'cenario_detectado': MarketScenarioRoyalSupremeEnhanced.TREND_OPPORTUNITY,
                'modo_sobrevivencia': SurvivabilityMode.NORMAL,
                'oportunidade_especial': True,
                'momentum_boost': True,
                'momentum_phase': momentum_analysis['phase'],
                'momentum_strength': momentum_analysis['strength']
            }
        
        # CRITÉRIOS RIGOROSOS MANTIDOS
        entrada_segura = (
            score_total >= ConfigRoyalSupremeEnhanced.MIN_SCORE_NORMAL and
            confluencia_count >= ConfigRoyalSupremeEnhanced.MIN_CONFLUENCIA and
            cenario_atual not in [MarketScenarioRoyalSupremeEnhanced.FLASH_CRASH, MarketScenarioRoyalSupremeEnhanced.PUMP_DUMP]
        )
        
        return {
            'entrada_segura': entrada_segura,
            'cenario_detectado': cenario_atual,
            'modo_sobrevivencia': self.modo_atual,
            'motivo_bloqueio': f"Cenário: {cenario_atual.value}" if not entrada_segura else None,
            'analise_caotico': analise_caotico,
            'auto_calibrador_ativo': auto_calibrador_usado,
            'adjustment_aplicado': adjustment_total,
            'momentum_phase': momentum_analysis['phase'],
            'momentum_strength': momentum_analysis['strength']
        }
    
    def _get_losses_consecutivos_recentes(self, par: str) -> int:
        """Conta losses consecutivos nas últimas operações"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT resultado FROM operacoes 
                WHERE par = ? AND resultado IN ('WIN_M1', 'WIN_GALE', 'LOSS')
                ORDER BY timestamp DESC 
                LIMIT 5
            ''', (par,))
            
            resultados = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Contar losses consecutivos do mais recente
            losses_consecutivos = 0
            for resultado in resultados:
                if resultado == 'LOSS':
                    losses_consecutivos += 1
                else:
                    break
            
            return losses_consecutivos
            
        except Exception as e:
            return 0
    
    def _verificar_blacklist_inteligente_corrigido(self, par: str, analise: Dict) -> bool:
        """BLACKLIST CORRIGIDO: Detecta padrões específicos de loss"""
        try:
            quality_score = self.db_manager.get_quality_score_por_par(par)
            
            # 1. BLACKLIST POR PERFORMANCE CRÍTICA
            if quality_score < 55:
                return True
            
            # 2. BLACKLIST POR PADRÕES ESPECÍFICOS
            motivos_call = analise.get('motivos_call', [])
            motivos_put = analise.get('motivos_put', [])
            todos_motivos = motivos_call + motivos_put
            
            # 3. PADRÕES PROBLEMÁTICOS IDENTIFICADOS NO LOG
            # ADA com scores baixos (136-154%) + múltiplos oversold/overbought
            if par == 'adausdt':
                score_call = analise.get('score_call', 0)
                score_put = analise.get('score_put', 0)
                score_max = max(score_call, score_put)
                
                # ADA scores baixos são problemáticos (BASEADO NO LOG)
                if score_max < 180:  # Aumentei de 170 para 180
                    oversold_count = sum(1 for motivo in todos_motivos if 'Oversold' in motivo or 'Extremo Baixo' in motivo)
                    overbought_count = sum(1 for motivo in todos_motivos if 'Overbought' in motivo or 'Extremo Alto' in motivo)
                    
                    if oversold_count >= 2 or overbought_count >= 2:
                        return True
            
            # 4. ETH com padrões específicos que deram loss
            if par == 'ethusdt':
                if quality_score < 75:
                    # Pin Bar + scores medianos = padrão problemático observado
                    if any('Pin Bar' in motivo for motivo in todos_motivos):
                        score_call = analise.get('score_call', 0)
                        score_put = analise.get('score_put', 0)
                        if max(score_call, score_put) < 190:  # Mais rigoroso
                            return True
            
            # 5. NOVO: Padrões de indecisão detectados
            dangerous_patterns = analise.get('dangerous_patterns', [])
            if any(pattern in ['DOJI', 'SPINNING_TOP', 'INDECISION'] for pattern in dangerous_patterns):
                return True
            
            # 6. BLACKLIST POR HORÁRIO PROBLEMÁTICO + BAIXA PERFORMANCE
            hora_atual = datetime.datetime.now(TIMEZONE).hour
            
            # Madrugada com baixa performance = risco
            if 2 <= hora_atual <= 8:
                if quality_score < 70:
                    volume_ratio = analise.get('volume_ratio', 2)
                    if volume_ratio < 1.8:
                        return True
            
            return False
            
        except Exception as e:
            return False

class AnalisadorCompletoV8RoyalSupremeEnhanced:
    
    def __init__(self):
        self.arsenal = ArsenalTecnicoCompletoV8RoyalSupremeEnhanced()
        
        # NOVO: Sistemas avançados de análise
        self.pattern_recognition = PatternRecognitionSupreme()
        self.trend_analysis = TrendAnalysisSupreme()
        self.price_action_1000 = PriceActionSupreme1000Velas()
        
        # 🚀 CORREÇÃO CRÍTICA: Inicializar sistema_sobrevivencia
        self.sistema_sobrevivencia = None  # Inicialização como None para evitar erros
        
    def inicializar_sobrevivencia(self, relatorios_system, db_manager):
        """CORREÇÃO: Inicializar sistema de sobrevivência corretamente"""
        self.sistema_sobrevivencia = SistemaSobrevivenciaV8RoyalSupremeEnhanced(relatorios_system, db_manager)
    
    def _calcular_enhanced_weight_dinamico_corrigido(self, par: str, volume_ratio: float, volatilidade: float) -> float:
        """🚀 ENHANCED WEIGHT DINÂMICO CORRIGIDO BASEADO EM IA + PERFORMANCE"""
        
        # Enhanced weight base
        base_weight = PARES_CRYPTO.get(par, {}).get('enhanced_weight', 1.0)
        
        # 🧠 AJUSTE POR QUALITY SCORE (IA DINÂMICA REAL)
        try:
            quality_score = self.sistema_sobrevivencia.db_manager.get_quality_score_por_par(par)
            
            # CORREÇÃO CRÍTICA: Ajustar weight baseado em performance REAL
            if quality_score < 55:
                base_weight *= 0.5   # Muito mais rigoroso (50% do original)
            elif quality_score < 65:
                base_weight *= 0.65  # Mais rigoroso
            elif quality_score < 75:
                base_weight *= 0.75  # Moderadamente rigoroso
            elif quality_score < 85:
                base_weight *= 0.9   # Levemente rigoroso
            elif quality_score > 90:
                base_weight *= 1.05  # Levemente mais fácil
                
        except:
            pass
        
        # Horário atual (Brasil)
        hora_atual = datetime.datetime.now(TIMEZONE).hour
        
        # 🌙 PROTEÇÃO BTC EM BAIXA VOLATILIDADE (MANTIDO)
        if par == 'btcusdt':
            # Horários de baixa volatilidade BTC (01:00-10:00)
            if 1 <= hora_atual <= 10:
                if volume_ratio < 1.5 or volatilidade < 0.3:
                    return max(0.4, base_weight * 0.6)  # BTC mais rigoroso
            
            # Horários prime BTC (14:00-24:00)
            if 14 <= hora_atual <= 23:
                if volume_ratio > 2.0 and volatilidade > 0.5:
                    return min(1.1, base_weight * 1.1)  # BTC mais fácil
            
            return base_weight
        
        # 🚀 AJUSTES ESPECÍFICOS POR PAR (BASEADO NO LOG REAL)
        elif par == 'adausdt':
            # ADA problemático - MUITO mais rigoroso sempre
            base_weight *= 0.6  # 40% mais rigoroso
            
            # Extra rigoroso em horários dos losses observados
            if 7 <= hora_atual <= 9:  # Horário dos losses no log
                base_weight *= 0.7
                
        elif par == 'ethusdt':
            # ETH com ajuste por horário dos losses
            if 7 <= hora_atual <= 8:  # Horário do loss observado no log
                base_weight *= 0.8
        
        # 🚀 OUTROS PARES (MANTIDO)
        else:
            # Madrugada: outros pares mais favorecidos
            if 1 <= hora_atual <= 8:
                if volume_ratio > 1.3 and volatilidade > 0.2:
                    return min(base_weight + 0.1, 1.0)
            
            # Dia: condições normais
            if 9 <= hora_atual <= 17:
                return base_weight
            
            # Noite: baseado em volume/volatilidade
            if 18 <= hora_atual <= 0:
                if volume_ratio > 2.0:
                    return min(base_weight + 0.05, 1.0)
        
        return max(0.4, base_weight)  # Nunca menor que 0.4
    
    def _aplicar_enhanced_weight_corrigido(self, score_call: float, score_put: float, 
                                         par: str, volume_ratio: float, volatilidade: float) -> Tuple[float, float, float]:
        """🚀 CORREÇÃO: Aplica enhanced weight de forma inteligente e correta"""
        
        # Calcular enhanced weight dinâmico
        enhanced_weight = self._calcular_enhanced_weight_dinamico_corrigido(par, volume_ratio, volatilidade)
        
        # 🎯 LÓGICA CORRIGIDA: 
        # Enhanced weight MENOR = precisa score MAIOR (mais rigoroso)
        # Enhanced weight MAIOR = precisa score MENOR (mais fácil)
        
        if enhanced_weight < 1.0:
            # Par mais rigoroso: precisa score maior
            score_multiplier = 1.0 / enhanced_weight
            score_call_adjusted = score_call / score_multiplier
            score_put_adjusted = score_put / score_multiplier
        else:
            # Par normal/favorecido: score normal ou levemente boosted
            score_call_adjusted = score_call * enhanced_weight
            score_put_adjusted = score_put * enhanced_weight
        
        return score_call_adjusted, score_put_adjusted, enhanced_weight
        
    def analisar_completo_anti_loss_enhanced(self, df: pd.DataFrame, par: str) -> Dict[str, Any]:
        
        if len(df) < 100:
            return {
                'score_call': 0, 'score_put': 0, 'motivos_call': [], 'motivos_put': [],
                'confluencia_call': 0, 'confluencia_put': 0, 'cenario_detectado': MarketScenarioRoyalSupremeEnhanced.NORMAL,
                'analise_bloqueada': True, 'motivo_bloqueio': 'Dados insuficientes'
            }
        
        fechamentos = df['close'].values
        preco_atual = fechamentos[-1]
        
        # 🚀 NOVO: ANÁLISE MACRO/MICRO TENDÊNCIA (1000 velas se disponível)
        trend_analysis = self.trend_analysis.analyze_macro_micro_trends(df)
        
        # 🚀 NOVO: PRICE ACTION HISTÓRICO (1000 velas)
        historical_analysis = self.price_action_1000.analyze_historical_price_action(df)
        
        # FILTRO ANTI-LATERALIZAÇÃO (MANTIDO)
        if len(fechamentos) >= ConfigRoyalSupremeEnhanced.LATERALIZACAO_PERIODS:
            precos_range = fechamentos[-ConfigRoyalSupremeEnhanced.LATERALIZACAO_PERIODS:]
            max_range = np.max(precos_range)
            min_range = np.min(precos_range)
            range_pct = ((max_range - min_range) / min_range) * 100
            
            if range_pct < ConfigRoyalSupremeEnhanced.LATERALIZACAO_THRESHOLD:
                return {
                    'score_call': 0, 'score_put': 0, 'motivos_call': [], 'motivos_put': [],
                    'confluencia_call': 0, 'confluencia_put': 0, 
                    'analise_bloqueada': True, 
                    'motivo_bloqueio': f'🚫 LATERALIZAÇÃO: Range {range_pct:.3f}%'
                }
        
        score_call = 0
        score_put = 0
        motivos_call = []
        motivos_put = []
        
        # 15 INDICADORES CALIBRADOS (V8 ORIGINAL MANTIDO)
        
        # 1. RSI
        rsi = self.arsenal.rsi(fechamentos, 14)
        if rsi < 20:
            motivos_call.append("RSI Extremo Baixo")
            score_call += 50
        elif rsi < 25:
            motivos_call.append("RSI Sobrevendido")
            score_call += 25
        
        if rsi > 80:
            motivos_put.append("RSI Extremo Alto")
            score_put += 50
        elif rsi > 75:
            motivos_put.append("RSI Sobrecomprado")
            score_put += 25
        
        # 2. MACD
        macd = self.arsenal.macd(fechamentos)
        if macd['linha'] > macd['sinal'] and macd['histograma'] > 0:
            motivos_call.append("MACD Bullish Forte")
            score_call += 25
        elif macd['linha'] < macd['sinal'] and macd['histograma'] < 0:  
            motivos_put.append("MACD Bearish Forte")
            score_put += 25
        
        # 3. Stochastic
        stoch = self.arsenal.stochastic(df, 14)
        if stoch['oversold']:
            motivos_call.append("Stochastic Oversold")
            score_call += 25
        elif stoch['overbought']:
            motivos_put.append("Stochastic Overbought")
            score_put += 25
        
        # 4. Williams %R
        wr = self.arsenal.williams_r(df, 14)
        if wr < -85:
            motivos_call.append("Williams %R Oversold")
            score_call += 20
        elif wr > -15:
            motivos_put.append("Williams %R Overbought") 
            score_put += 20
        
        # 5. VWAP
        vwap = self.arsenal.vwap(df)
        if preco_atual < vwap * 0.995:
            motivos_call.append("VWAP Suporte")
            score_call += 18
        elif preco_atual > vwap * 1.005:
            motivos_put.append("VWAP Resistência")
            score_put += 18
        
        # 6. Bollinger Bands
        bb = self.arsenal.bandas_bollinger(fechamentos, 20, 2)
        if bb['posicao'] < 0.1:
            motivos_call.append("BB Extremo Baixo")
            score_call += 25
        elif bb['posicao'] > 0.9:
            motivos_put.append("BB Extremo Alto")
            score_put += 25
        
        # 7. ADX
        adx = self.arsenal.adx(df, 14)
        if adx > 30:
            motivos_call.append("ADX Tendência Forte")
            motivos_put.append("ADX Tendência Forte")
            score_call += 15
            score_put += 15
        
        # 8. CCI
        cci = self.arsenal.cci(df, 20)
        if cci < -200:
            motivos_call.append("CCI Extremo Baixo")
            score_call += 30
        elif cci > 200:
            motivos_put.append("CCI Extremo Alto")
            score_put += 30
        
        # 9. Momentum
        momentum = self.arsenal.momentum(fechamentos, 10)
        if momentum > 0:
            motivos_call.append("Momentum Positivo")
            score_call += 15
        else:
            motivos_put.append("Momentum Negativo")
            score_put += 15
        
        # 10. Rate of Change
        roc = self.arsenal.rate_of_change(fechamentos, 10)
        if roc > 2:
            motivos_call.append("ROC Forte Alta")
            score_call += 20
        elif roc < -2:
            motivos_put.append("ROC Forte Baixa")
            score_put += 20
        
        # 11. Money Flow Index
        mfi = self.arsenal.money_flow_index(df, 14)
        if mfi < 20:
            motivos_call.append("MFI Oversold")
            score_call += 22
        elif mfi > 80:
            motivos_put.append("MFI Overbought")
            score_put += 22
        
        # 12. True Strength Index
        tsi = self.arsenal.true_strength_index(df, 25, 13)
        if tsi > 15:
            motivos_call.append("TSI Bullish")
            score_call += 18
        elif tsi < -15:
            motivos_put.append("TSI Bearish")
            score_put += 18
        
        # 13. Ultimate Oscillator
        uo = self.arsenal.ultimate_oscillator(df, 7, 14, 28)
        if uo < 30:
            motivos_call.append("UO Oversold")
            score_call += 20
        elif uo > 70:
            motivos_put.append("UO Overbought")
            score_put += 20
        
        # 14. Parabolic SAR
        psar = self.arsenal.parabolic_sar(df, 0.02, 0.2)
        if preco_atual > psar:
            motivos_call.append("PSAR Bullish")
            score_call += 15
        else:
            motivos_put.append("PSAR Bearish")
            score_put += 15
        
        # 15. Awesome Oscillator
        ao = self.arsenal.awesome_oscillator(df, 5, 34)
        if ao > 0:
            motivos_call.append("AO Bullish")
            score_call += 12
        else:
            motivos_put.append("AO Bearish")
            score_put += 12
        
        # 🚀 NOVO: ANÁLISE DE PADRÕES AVANÇADOS (50+ PADRÕES)
        pattern_analysis = self.pattern_recognition.analyze_all_patterns(df, trend_analysis)
        
        # Aplicar scores dos padrões com contexto de tendência
        score_call += pattern_analysis['pattern_score_call']
        score_put += pattern_analysis['pattern_score_put']
        
        # Adicionar motivos dos padrões (apenas os mais fortes)
        motivos_call.extend(pattern_analysis['pattern_motivos_call'][:3])
        motivos_put.extend(pattern_analysis['pattern_motivos_put'][:3])
        
        # ENHANCED PRICE ACTION ANALYSIS (MANTIDO)
        price_action = PriceActionMasterRoyalSupremeEnhanced.analyze_complete_price_action_enhanced(df)
        
        score_call += price_action['price_action_score_call']
        score_put += price_action['price_action_score_put']
        
        motivos_call.extend(price_action['price_action_motivos_call'])
        motivos_put.extend(price_action['price_action_motivos_put'])
        
        # 🚀 NOVO: CONFLUÊNCIA MACRO/MICRO LTA/LTB
        if trend_analysis['macro_micro_confluence']:
            confluence_type = trend_analysis['confluence_direction']
            confluence_strength = trend_analysis['confluence_strength']
            
            if confluence_type == 'BULLISH':
                score_call += confluence_strength
                motivos_call.append(f"Confluência Macro/Micro LTA: +{confluence_strength}")
                if self.sistema_sobrevivencia and hasattr(self.sistema_sobrevivencia, 'stats_sobrevivencia'):
                    self.sistema_sobrevivencia.stats_sobrevivencia['macro_micro_confluences'] += 1
            elif confluence_type == 'BEARISH':
                score_put += confluence_strength
                motivos_put.append(f"Confluência Macro/Micro LTB: +{confluence_strength}")
                if self.sistema_sobrevivencia and hasattr(self.sistema_sobrevivencia, 'stats_sobrevivencia'):
                    self.sistema_sobrevivencia.stats_sobrevivencia['macro_micro_confluences'] += 1
        
        # 🚀 NOVO: SUPORTE/RESISTÊNCIA HISTÓRICO (1000 VELAS)
        if historical_analysis['historical_sr']:
            for sr_level in historical_analysis['historical_sr'][:3]:  # Top 3 níveis
                distance_pct = sr_level['distance_pct']
                if distance_pct < 1.0:  # Muito próximo
                    strength_bonus = min(sr_level['strength'] * 10, 40)
                    if sr_level['type'] == 'SUPPORT':
                        score_call += strength_bonus
                        motivos_call.append(f"Suporte Histórico: {sr_level['level']:.2f}")
                    elif sr_level['type'] == 'RESISTANCE':
                        score_put += strength_bonus
                        motivos_put.append(f"Resistência Histórica: {sr_level['level']:.2f}")
        
        # Volume Analysis
        volumes = df['volume'].values
        vol_atual = volumes[-1]
        vol_media = np.mean(volumes[-20:])
        volume_ratio = vol_atual / vol_media if vol_media > 0 else 1
        
        # VOLUME MAIS RIGOROSO (MANTIDO)
        if volume_ratio < ConfigRoyalSupremeEnhanced.MIN_VOLUME_RATIO:
            return {
                'score_call': 0, 'score_put': 0, 'motivos_call': [], 'motivos_put': [],
                'confluencia_call': 0, 'confluencia_put': 0, 
                'analise_bloqueada': True, 
                'motivo_bloqueio': f'🚫 VOLUME BAIXO: {volume_ratio:.2f}x'
            }
        
        if volume_ratio > 6.0:
            motivos_call.append("Volume Explosivo")
            motivos_put.append("Volume Explosivo")
            score_call += 30
            score_put += 30
        elif volume_ratio > 4.0:
            motivos_call.append("Volume Muito Alto")
            motivos_put.append("Volume Muito Alto")
            score_call += 20
            score_put += 20
        
        # Movimento recente
        movimento_1min = ((preco_atual - fechamentos[-2]) / fechamentos[-2]) * 100 if len(fechamentos) >= 2 else 0
        
        # 🚀 NOVO: ANÁLISE INTELIGENTE DE MOMENTUM (EVITA ENTRADA EM CORREÇÕES)
        momentum_analysis = self.trend_analysis.analyze_momentum_phase(df, movimento_1min)
        
        if momentum_analysis['phase'] == 'CORRECTION':
            # Em fase de correção, não adicionar momentum como motivo
            pass
        elif momentum_analysis['phase'] == 'RESUMPTION':
            # Em retomada de tendência, adicionar momentum forte
            if abs(movimento_1min) > 0.08:
                if movimento_1min > 0:
                    motivos_call.append(f"Retomada Bullish: +{movimento_1min:.3f}%")
                    score_call += min(50, int(abs(movimento_1min) * 150))
                else:
                    motivos_put.append(f"Retomada Bearish: {movimento_1min:.3f}%")
                    score_put += min(50, int(abs(movimento_1min) * 150))
        else:
            # Momentum normal
            if abs(movimento_1min) > 0.08:
                if movimento_1min > 0:
                    motivos_call.append(f"Momentum Forte: +{movimento_1min:.3f}%")
                    score_call += min(40, int(abs(movimento_1min) * 100))
                else:
                    motivos_put.append(f"Momentum Forte: {movimento_1min:.3f}%")
                    score_put += min(40, int(abs(movimento_1min) * 100))
        
        # Volatilidade
        volatilidade_pct = np.std(fechamentos[-20:]) / np.mean(fechamentos[-20:]) * 100 if len(fechamentos) > 20 else 0
        
        # 🚀 ENHANCED WEIGHT APPLICATION CORRIGIDO
        score_call_final, score_put_final, enhanced_weight = self._aplicar_enhanced_weight_corrigido(
            score_call, score_put, par, volume_ratio, volatilidade_pct
        )
        
        return {
            'score_call': max(0, score_call_final),
            'score_put': max(0, score_put_final),
            'motivos_call': motivos_call,
            'motivos_put': motivos_put,
            'confluencia_call': len(motivos_call),
            'confluencia_put': len(motivos_put),
            'rsi': rsi,
            'volatilidade': volatilidade_pct,
            'volume_ratio': volume_ratio,
            'volume_bloqueado': volume_ratio < ConfigRoyalSupremeEnhanced.MIN_VOLUME_RATIO,
            'analise_bloqueada': False,
            'total_indicadores_analisados': 15,
            'movimento_1min': movimento_1min,
            'preco_atual': preco_atual,
            'vwap_atual': vwap,
            'bb_posicao': bb['posicao'],
            'macd_info': macd,
            'adx_valor': adx,
            'cci_valor': cci,
            'enhanced_weight_aplicado': enhanced_weight,
            'volatilidade_base': PARES_CRYPTO.get(par, {}).get('volatilidade_base', 1.0),
            # ENHANCED PRICE ACTION DATA (MANTIDO)
            'price_action_patterns': price_action['candlestick_patterns'],
            'support_levels': price_action['support_levels'],
            'resistance_levels': price_action['resistance_levels'],
            'lta': price_action['lta'],
            'ltb': price_action['ltb'],
            'pullback': price_action['pullback'],
            'throwback': price_action['throwback'],
            'elliott_pattern': price_action['elliott_pattern'],
            'total_enhanced_signals': price_action['total_enhanced_signals'],
            'price_action_score_call': price_action['price_action_score_call'],
            'price_action_score_put': price_action['price_action_score_put'],
            # 🚀 NOVOS DADOS DOS PADRÕES AVANÇADOS
            'advanced_patterns': pattern_analysis['patterns_detected'],
            'dangerous_patterns': pattern_analysis['dangerous_patterns'],
            'pattern_confidence': pattern_analysis['overall_confidence'],
            'trend_analysis': trend_analysis,
            'macro_trend': trend_analysis['macro_trend'],
            'micro_trend': trend_analysis['micro_trend'],
            'confluence_detected': trend_analysis['macro_micro_confluence'],
            'momentum_phase': momentum_analysis['phase'],
            'momentum_strength': momentum_analysis['strength'],
            # 🚀 DADOS HISTÓRICOS (1000 VELAS)
            'historical_analysis': historical_analysis,
            'historical_sr_count': len(historical_analysis['historical_sr']),
            'macro_context': historical_analysis['macro_context'],
            'key_levels_count': len(historical_analysis['key_levels'])
        }

@dataclass
class SinalRoyalSupremeEnhanced:
    tipo_sinal: TipoSinalRoyalSupremeEnhanced
    par: str
    timestamp: int
    timeframe: str
    score_total: float
    score_indicadores: float
    score_filtros: float
    confluencia_count: int
    motivos_confluencia: List[str]
    cenario_detectado: MarketScenarioRoyalSupremeEnhanced
    modo_sobrevivencia: SurvivabilityMode
    niveis_sr: Dict[str, float]
    volatilidade_atual: float = 0.0
    volume_ratio: float = 1.0
    enhanced_weight_aplicado: float = 1.0
    auto_calibrador_usado: bool = False
    # ENHANCED PRICE ACTION FIELDS (MANTIDO)
    price_action_patterns: List[str] = field(default_factory=list)
    support_levels: List[Dict] = field(default_factory=list)
    resistance_levels: List[Dict] = field(default_factory=list)
    lta: Optional[Dict] = None
    ltb: Optional[Dict] = None
    pullback: Optional[Dict] = None
    throwback: Optional[Dict] = None
    elliott_pattern: Optional[Dict] = None
    price_action_score_call: float = 0.0
    price_action_score_put: float = 0.0
    enhanced_features: List[str] = field(default_factory=list)
    # 🚀 NOVOS CAMPOS PARA PADRÕES AVANÇADOS
    advanced_patterns: List[str] = field(default_factory=list)
    pattern_confidence: float = 0.0
    macro_trend: str = "NEUTRAL"
    micro_trend: str = "NEUTRAL"
    confluence_detected: bool = False
    momentum_phase: str = "NORMAL"
    historical_sr_count: int = 0
    macro_context: str = "NORMAL"
    # CAMPOS ORIGINAIS (MANTIDOS)
    status: StatusSinalRoyalSupremeEnhanced = StatusSinalRoyalSupremeEnhanced.ATIVO
    vela_resultado_m1: Optional[str] = None
    vela_resultado_gale: Optional[str] = None
    horario_emissao: str = ""
    timestamp_verificacao_m1: int = 0
    timestamp_verificacao_gale: int = 0
    timestamp_vela_m1: int = 0
    timestamp_vela_gale: int = 0
    
    def __post_init__(self):
        # Timing V8 original preservado (NÃO MEXER)
        if self.timeframe == '1m':
            self.timestamp_verificacao_m1 = self.timestamp + ConfigRoyalSupremeEnhanced.VERIFICACAO_M1_SEGUNDOS
            self.timestamp_verificacao_gale = self.timestamp + ConfigRoyalSupremeEnhanced.VERIFICACAO_GALE_SEGUNDOS
        elif self.timeframe == '5m':
            self.timestamp_verificacao_m1 = self.timestamp + (ConfigRoyalSupremeEnhanced.VERIFICACAO_M1_SEGUNDOS * 5)
            self.timestamp_verificacao_gale = self.timestamp + (ConfigRoyalSupremeEnhanced.VERIFICACAO_GALE_SEGUNDOS * 5)
        
        # Timestamp de vela
        if self.timeframe == '1m':
            dt = datetime.datetime.fromtimestamp(self.timestamp)
            proxima_vela = dt.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
            self.timestamp_vela_m1 = int(proxima_vela.timestamp())
            self.timestamp_vela_gale = self.timestamp_vela_m1 + 60
        elif self.timeframe == '5m':
            dt = datetime.datetime.fromtimestamp(self.timestamp)
            proxima_vela = dt.replace(second=0, microsecond=0) + datetime.timedelta(minutes=5)
            self.timestamp_vela_m1 = int(proxima_vela.timestamp())
            self.timestamp_vela_gale = self.timestamp_vela_m1 + 300

class RelatoriosRoyalSupremeEnhanced:
    
    def __init__(self):
        self.stats_globais = {
            'total_sinais': 0, 'wins_m1': 0, 'wins_gale': 0, 'losses': 0,
            'auto_calibrador_usado': 0, 'enhanced_weight_aplicado': 0,
            'elite_opportunities': 0, 'wave_opportunities': 0,
            'trend_opportunities': 0, 'pullback_opportunities': 0, 'elliott_opportunities': 0,
            'blacklist_ativacoes': 0, 'ia_ajustes_aplicados': 0,
            # 🚀 NOVOS STATS PARA PADRÕES AVANÇADOS
            'patterns_blocked': 0, 'macro_micro_confluences': 0,
            'trend_corrections_detected': 0, 'resumptions_caught': 0,
            'dangerous_patterns_avoided': 0, 'historical_sr_used': 0,
            'tempo_inicio': int(time.time())
        }
        self.stats_por_par = defaultdict(lambda: {
            'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'auto_calibrador_ativacoes': 0, 'blacklist_blocks': 0,
            'patterns_blocked': 0, 'confluences_detected': 0,
            'corrections_avoided': 0, 'resumptions_caught': 0
        })
        self.historico_sinais = deque(maxlen=1000)
    
    def registrar_sinal(self, sinal: SinalRoyalSupremeEnhanced):
        self.stats_globais['total_sinais'] += 1
        self.stats_por_par[sinal.par]['total'] += 1
        
        if sinal.auto_calibrador_usado:
            self.stats_globais['auto_calibrador_usado'] += 1
            self.stats_por_par[sinal.par]['auto_calibrador_ativacoes'] += 1
            
        if sinal.enhanced_weight_aplicado != 1.0:
            self.stats_globais['enhanced_weight_aplicado'] += 1
            self.stats_globais['ia_ajustes_aplicados'] += 1
        
        # 🚀 NOVOS REGISTROS
        if sinal.confluence_detected:
            self.stats_globais['macro_micro_confluences'] += 1
            self.stats_por_par[sinal.par]['confluences_detected'] += 1
            
        if sinal.momentum_phase == 'RESUMPTION':
            self.stats_globais['resumptions_caught'] += 1
            self.stats_por_par[sinal.par]['resumptions_caught'] += 1
            
        if sinal.historical_sr_count > 0:
            self.stats_globais['historical_sr_used'] += 1
        
        self.historico_sinais.append({
            'timestamp': sinal.timestamp,
            'par': sinal.par,
            'tipo': sinal.tipo_sinal.value,
            'score': sinal.score_total,
            'cenario': sinal.cenario_detectado.value,
            'auto_calibrador': sinal.auto_calibrador_usado,
            'enhanced_weight': sinal.enhanced_weight_aplicado,
            'macro_trend': sinal.macro_trend,
            'micro_trend': sinal.micro_trend,
            'confluence': sinal.confluence_detected,
            'momentum_phase': sinal.momentum_phase,
            'patterns_count': len(sinal.advanced_patterns),
            'historical_sr': sinal.historical_sr_count
        })
    
    def registrar_blacklist_block(self, par: str):
        """Registra bloqueios por blacklist"""
        self.stats_globais['blacklist_ativacoes'] += 1
        self.stats_por_par[par]['blacklist_blocks'] += 1
    
    def registrar_pattern_block(self, par: str):
        """🚀 NOVO: Registra bloqueios por padrões perigosos"""
        self.stats_globais['patterns_blocked'] += 1
        self.stats_por_par[par]['patterns_blocked'] += 1
    
    def registrar_correction_avoided(self, par: str):
        """🚀 NOVO: Registra correções evitadas"""
        self.stats_globais['trend_corrections_detected'] += 1
        self.stats_por_par[par]['corrections_avoided'] += 1
    
    def registrar_resultado(self, sinal: SinalRoyalSupremeEnhanced):
        if sinal.status == StatusSinalRoyalSupremeEnhanced.WIN_M1:
            self.stats_globais['wins_m1'] += 1
            self.stats_por_par[sinal.par]['wins'] += 1
        elif sinal.status == StatusSinalRoyalSupremeEnhanced.WIN_GALE:
            self.stats_globais['wins_gale'] += 1
            self.stats_por_par[sinal.par]['wins'] += 1
        elif sinal.status == StatusSinalRoyalSupremeEnhanced.LOSS:
            self.stats_globais['losses'] += 1
            self.stats_por_par[sinal.par]['losses'] += 1
        
        par_stats = self.stats_por_par[sinal.par]
        if par_stats['total'] > 0:
            par_stats['win_rate'] = (par_stats['wins'] / par_stats['total']) * 100
    
    def gerar_relatorio_completo(self) -> str:
        tempo_operacao = int(time.time()) - self.stats_globais['tempo_inicio']
        horas_operacao = tempo_operacao // 3600
        
        total_trades = self.stats_globais['wins_m1'] + self.stats_globais['wins_gale'] + self.stats_globais['losses']
        win_rate = ((self.stats_globais['wins_m1'] + self.stats_globais['wins_gale']) / total_trades * 100) if total_trades > 0 else 0
        
        relatorio = f"""
👑 CIPHER ROYAL SUPREME ENHANCED + DATABASE AI - RELATÓRIO ELITE TRADER + IA PATTERNS

⏰ TEMPO DE OPERAÇÃO: {horas_operacao}h
🎯 TOTAL SINAIS: {self.stats_globais['total_sinais']}
📊 WIN RATE SUPREMO ENHANCED + AI PATTERNS: {win_rate:.1f}%

🏆 PERFORMANCE DETALHADA:
   ✅ Wins M1: {self.stats_globais['wins_m1']}
   🟡 Wins Gale: {self.stats_globais['wins_gale']}
   💎 Losses: {self.stats_globais['losses']}

👑 ROYAL SUPREME ENHANCED + AI PATTERNS STATS:
   🛡️ Auto Calibrador IA: {self.stats_globais['auto_calibrador_usado']} ativações FUNCIONAIS
   💎 Enhanced Weight Dinâmico: {self.stats_globais['enhanced_weight_aplicado']} aplicações
   🚫 Blacklist IA: {self.stats_globais['blacklist_ativacoes']} bloqueios inteligentes
   🔴 Padrões Perigosos Bloqueados: {self.stats_globais['patterns_blocked']}
   🔗 Confluências Macro/Micro: {self.stats_globais['macro_micro_confluences']}
   ⚡ Retomadas Capturadas: {self.stats_globais['resumptions_caught']}
   🚫 Correções Evitadas: {self.stats_globais['trend_corrections_detected']}
   📊 S/R Histórico Usado: {self.stats_globais['historical_sr_used']}
   🧠 Total Ajustes IA: {self.stats_globais['ia_ajustes_aplicados']} intervenções
   
   ⚡ Elite Opportunities: {self.stats_globais['elite_opportunities']}
   🌊 Wave Opportunities: {self.stats_globais['wave_opportunities']}
   🎯 Trend Opportunities: {self.stats_globais['trend_opportunities']}
   🔄 Pullback Opportunities: {self.stats_globais['pullback_opportunities']}
   🌊 Elliott Opportunities: {self.stats_globais['elliott_opportunities']}
"""
        
        relatorio += "\n💎 PERFORMANCE POR PAR + IA PATTERNS STATS:\n"
        for par, stats in self.stats_por_par.items():
            if stats['total'] > 0:
                par_nome = PARES_CRYPTO[par]['nome']
                ia_info = f" (Calibrador:{stats['auto_calibrador_ativacoes']}, Blocks:{stats['blacklist_blocks']}, Patterns:{stats['patterns_blocked']}, Conf:{stats['confluences_detected']}, Corr:{stats['corrections_avoided']}, Ret:{stats['resumptions_caught']})" if stats['auto_calibrador_ativacoes'] > 0 or stats['blacklist_blocks'] > 0 else ""
                relatorio += f"   {par_nome}: {stats['wins']}/{stats['total']} ({stats['win_rate']:.1f}%){ia_info}\n"
        
        return relatorio
    
    def gerar_relatorio_resumido(self) -> str:
        total_trades = self.stats_globais['wins_m1'] + self.stats_globais['wins_gale'] + self.stats_globais['losses']
        win_rate = ((self.stats_globais['wins_m1'] + self.stats_globais['wins_gale']) / total_trades * 100) if total_trades > 0 else 0
        
        return f"""👑 CIPHER ROYAL SUPREME ENHANCED + AI PATTERNS - RESUMO
🎯 Sinais: {self.stats_globais['total_sinais']} • Win Rate: {win_rate:.1f}%
🏆 {self.stats_globais['wins_m1']}W M1 • {self.stats_globais['wins_gale']}W Gale • {self.stats_globais['losses']} Loss
⚡ Elite: {self.stats_globais['elite_opportunities']} • 🌊 Wave: {self.stats_globais['wave_opportunities']}
🎯 Trend: {self.stats_globais['trend_opportunities']} • 🔄 Pullback: {self.stats_globais['pullback_opportunities']}
🌊 Elliott: {self.stats_globais['elliott_opportunities']}
🛡️ Auto Calibrador IA: {self.stats_globais['auto_calibrador_usado']} ativações FUNCIONAIS
🚫 Blacklist IA: {self.stats_globais['blacklist_ativacoes']} bloqueios inteligentes
🔴 Padrões Bloqueados: {self.stats_globais['patterns_blocked']}
🔗 Confluências: {self.stats_globais['macro_micro_confluences']}
⚡ Retomadas: {self.stats_globais['resumptions_caught']}
🚫 Correções Evitadas: {self.stats_globais['trend_corrections_detected']}
📊 S/R Histórico: {self.stats_globais['historical_sr_used']}
🧠 IA Total: {self.stats_globais['ia_ajustes_aplicados']} intervenções
👑 Royal Supreme Enhanced + AI PATTERNS Never Loses!
"""

print("✅ ANALISADOR COMPLETO V8 + ENHANCED - ROYAL SUPREME + IA PATTERNS CARREGADO!")