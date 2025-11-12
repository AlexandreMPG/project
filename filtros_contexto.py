#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🛡️ FILTROS CONTEXTO - CIPHER ROYAL SUPREME ENHANCED + AI ANTI-LOSS 🛡️
💎 FILTROS TÉCNICOS INTELIGENTES PARA DETECÇÃO DE CENÁRIOS PERIGOSOS
🔥 ANTI-EXAUSTÃO + S/R REAL + ANTI-ARMADILHA + ANTI-LATERAL + ANTI-GAP
🎯 EVITA LOSSES NAS SITUAÇÕES DAS "CRUZETAS" IDENTIFICADAS
🚀 CORREÇÃO: IA SUPERVISORA QUE ADAPTA AO INVÉS DE BLOQUEAR
"""

import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class FiltroResult:
    """Resultado de um filtro específico"""
    bloqueado: bool
    motivo: str
    confianca: float
    dados_extras: Dict[str, Any]
    # 🚀 NOVO: Campo para sugestão de reversão
    sugerir_reversao: bool = False
    tipo_sugerido: Optional[str] = None
    score_boost_reversao: float = 0.0

@dataclass
class ContextAnalysis:
    """Análise completa de contexto"""
    entrada_segura: bool
    filtros_aplicados: List[str]
    motivos_bloqueio: List[str]
    score_confianca: float
    ajuste_score: float
    dados_contexto: Dict[str, Any]
    # 🚀 NOVO: Campos para IA supervisora
    ia_sugestao: Optional[str] = None
    tipo_sugerido_ia: Optional[str] = None
    motivo_ia: str = ""
    score_boost_ia: float = 0.0

class FiltroAntiExaustao:
    """🔴 FILTRO ANTI-EXAUSTÃO: Detecta movimentos esgotados"""
    
    @staticmethod
    def detectar_exaustao(df: pd.DataFrame, tipo_sinal: str, rsi: float) -> FiltroResult:
        """Detecta sinais de exaustão no movimento"""
        if len(df) < 10:
            return FiltroResult(False, "", 0.0, {})
        
        closes = df['close'].values[-10:]
        volumes = df['volume'].values[-10:]
        highs = df['high'].values[-10:]
        lows = df['low'].values[-10:]
        
        # 1. SEQUÊNCIA DE VELAS DA MESMA COR
        velas_mesma_cor = 0
        for i in range(1, len(closes)):
            if 'CALL' in tipo_sinal:
                if closes[i] > closes[i-1]:
                    velas_mesma_cor += 1
                else:
                    break
            else:
                if closes[i] < closes[i-1]:
                    velas_mesma_cor += 1
                else:
                    break
        
        # 2. RSI EXTREMO
        rsi_extremo = (rsi > 80 and 'CALL' in tipo_sinal) or (rsi < 20 and 'PUT' in tipo_sinal)
        
        # 3. VOLUME ANÔMALO (Spike seguido de queda)
        if len(volumes) >= 5:
            vol_atual = volumes[-1]
            vol_media = np.mean(volumes[-5:-1])
            vol_spike = vol_atual > vol_media * 3.0
            
            # Volume diminuindo após spike
            vol_diminuindo = volumes[-1] < volumes[-2] < volumes[-3]
        else:
            vol_spike = False
            vol_diminuindo = False
        
        # 4. SOMBRAS LONGAS (Rejeição)
        if len(df) >= 1:
            ultima_vela = df.iloc[-1]
            o, h, l, c = ultima_vela['open'], ultima_vela['high'], ultima_vela['low'], ultima_vela['close']
            
            range_total = h - l
            if range_total > 0:
                sombra_superior = (h - max(o, c)) / range_total
                sombra_inferior = (min(o, c) - l) / range_total
                
                # Sombra longa no topo para CALL ou base para PUT
                sombra_rejeicao = (
                    ('CALL' in tipo_sinal and sombra_superior > 0.6) or
                    ('PUT' in tipo_sinal and sombra_inferior > 0.6)
                )
            else:
                sombra_rejeicao = False
        else:
            sombra_rejeicao = False
        
        # DECISÃO DE BLOQUEIO
        sinais_exaustao = 0
        motivos = []
        
        if velas_mesma_cor >= 4:
            sinais_exaustao += 2
            motivos.append(f"{velas_mesma_cor} velas consecutivas")
        
        if rsi_extremo:
            sinais_exaustao += 2
            motivos.append(f"RSI extremo: {rsi:.1f}")
        
        if vol_spike and vol_diminuindo:
            sinais_exaustao += 1
            motivos.append("Volume spike + queda")
        
        if sombra_rejeicao:
            sinais_exaustao += 2
            motivos.append("Sombra longa (rejeição)")
        
        # 🚀 CORREÇÃO: Em vez de bloquear, sugerir reversão em exaustão extrema
        if sinais_exaustao >= 4:  # Era 3, agora 4 (mais rigoroso)
            tipo_sugerido = 'PUT' if 'CALL' in tipo_sinal else 'CALL'
            return FiltroResult(
                bloqueado=False,  # NÃO bloqueia mais
                motivo="",
                confianca=min(sinais_exaustao / 5.0, 1.0),
                dados_extras={
                    'velas_consecutivas': velas_mesma_cor,
                    'rsi': rsi,
                    'sinais_exaustao': sinais_exaustao,
                    'volume_spike': vol_spike,
                    'sombra_rejeicao': sombra_rejeicao
                },
                sugerir_reversao=True,
                tipo_sugerido=tipo_sugerido,
                score_boost_reversao=30.0
            )
        
        # Bloquear apenas em casos extremos
        bloqueado = sinais_exaustao >= 5
        confianca = min(sinais_exaustao / 5.0, 1.0)
        motivo = f"Exaustão extrema: {', '.join(motivos)}" if bloqueado else ""
        
        return FiltroResult(
            bloqueado=bloqueado,
            motivo=motivo,
            confianca=confianca,
            dados_extras={
                'velas_consecutivas': velas_mesma_cor,
                'rsi': rsi,
                'sinais_exaustao': sinais_exaustao,
                'volume_spike': vol_spike,
                'sombra_rejeicao': sombra_rejeicao
            }
        )

class FiltroSupporteResistenciaReal:
    """📊 FILTRO S/R REAL: IA SUPERVISORA QUE ADAPTA ESTRATÉGIA"""
    
    @staticmethod
    def analisar_sr_real(df: pd.DataFrame, tipo_sinal: str, lookback: int = 50) -> FiltroResult:
        """🚀 CORREÇÃO PRINCIPAL: Analisa S/R e sugere estratégia inteligente"""
        if len(df) < lookback:
            return FiltroResult(False, "", 0.0, {})
        
        highs = df['high'].values[-lookback:]
        lows = df['low'].values[-lookback:]
        current_price = df['close'].iloc[-1]
        
        # DETECTAR NÍVEIS DE S/R COM MÚLTIPLOS TOQUES
        support_levels = []
        resistance_levels = []
        
        # Buscar pivôs
        for i in range(2, len(lows) - 2):
            # Suporte
            if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                support_levels.append(lows[i])
            
            # Resistência
            if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                resistance_levels.append(highs[i])
        
        # AGRUPAR NÍVEIS PRÓXIMOS E CONTAR TOQUES
        def agrupar_niveis(levels, tolerance=0.002):
            if not levels:
                return []
            
            levels_sorted = sorted(levels)
            grupos = []
            grupo_atual = [levels_sorted[0]]
            
            for level in levels_sorted[1:]:
                if abs(level - grupo_atual[-1]) / grupo_atual[-1] <= tolerance:
                    grupo_atual.append(level)
                else:
                    grupos.append(grupo_atual)
                    grupo_atual = [level]
            grupos.append(grupo_atual)
            
            # Calcular nível médio e força de cada grupo
            niveis_fortes = []
            for grupo in grupos:
                nivel_medio = np.mean(grupo)
                toques = len(grupo)
                if toques >= 2:  # Mínimo 2 toques
                    niveis_fortes.append({
                        'nivel': nivel_medio,
                        'toques': toques,
                        'forca': toques * 10,
                        'distancia': abs(current_price - nivel_medio) / current_price
                    })
            
            return niveis_fortes
        
        suportes_fortes = agrupar_niveis(support_levels)
        resistencias_fortes = agrupar_niveis(resistance_levels)
        
        # 🚀 NOVA LÓGICA IA SUPERVISORA
        return FiltroSupporteResistenciaReal._decidir_estrategia_inteligente(
            tipo_sinal, current_price, suportes_fortes, resistencias_fortes
        )
    
    @staticmethod
    def _decidir_estrategia_inteligente(tipo_sinal: str, current_price: float, 
                                      suportes_fortes: List[Dict], 
                                      resistencias_fortes: List[Dict]) -> FiltroResult:
        """🧠 NOVA LÓGICA: IA que adapta estratégia ao invés de bloquear"""
        
        dados_extras = {
            'suportes_fortes': len(suportes_fortes),
            'resistencias_fortes': len(resistencias_fortes),
            'current_price': current_price
        }
        
        # 🚀 CASO 1: CALL próximo de RESISTÊNCIA FORTE → SUGERIR PUT (REVERSÃO)
        if 'CALL' in tipo_sinal:
            for res in resistencias_fortes:
                if res['distancia'] < 0.008 and res['toques'] >= 3:  # Muito próximo e forte
                    dados_extras['nivel_conflitante'] = res
                    dados_extras['decisao_ia'] = 'REVERSAO_RESISTENCIA'
                    
                    return FiltroResult(
                        bloqueado=False,  # NÃO bloqueia
                        motivo="",
                        confianca=min(res['forca'] / 50.0, 1.0),
                        dados_extras=dados_extras,
                        sugerir_reversao=True,
                        tipo_sugerido='PUT',
                        score_boost_reversao=45.0  # Boost forte para reversão
                    )
            
            # CALL próximo de SUPORTE FORTE → CONFIRMAR (BOUNCE)
            for sup in suportes_fortes:
                if sup['distancia'] < 0.005 and sup['toques'] >= 3:
                    dados_extras['nivel_favoravel'] = sup
                    dados_extras['decisao_ia'] = 'CONFIRMACAO_SUPORTE'
                    
                    return FiltroResult(
                        bloqueado=False,
                        motivo="",
                        confianca=0.8,
                        dados_extras=dados_extras,
                        sugerir_reversao=False,  # Confirma original
                        score_boost_reversao=20.0  # Boost para confirmação
                    )
        
        # 🚀 CASO 2: PUT próximo de SUPORTE FORTE → SUGERIR CALL (BOUNCE)
        elif 'PUT' in tipo_sinal:
            for sup in suportes_fortes:
                if sup['distancia'] < 0.008 and sup['toques'] >= 3:  # Muito próximo e forte
                    dados_extras['nivel_conflitante'] = sup
                    dados_extras['decisao_ia'] = 'REVERSAO_SUPORTE'
                    
                    return FiltroResult(
                        bloqueado=False,  # NÃO bloqueia
                        motivo="",
                        confianca=min(sup['forca'] / 50.0, 1.0),
                        dados_extras=dados_extras,
                        sugerir_reversao=True,
                        tipo_sugerido='CALL',
                        score_boost_reversao=45.0  # Boost forte para reversão
                    )
            
            # PUT próximo de RESISTÊNCIA FORTE → CONFIRMAR (REJEIÇÃO)
            for res in resistencias_fortes:
                if res['distancia'] < 0.005 and res['toques'] >= 3:
                    dados_extras['nivel_favoravel'] = res
                    dados_extras['decisao_ia'] = 'CONFIRMACAO_RESISTENCIA'
                    
                    return FiltroResult(
                        bloqueado=False,
                        motivo="",
                        confianca=0.8,
                        dados_extras=dados_extras,
                        sugerir_reversao=False,  # Confirma original
                        score_boost_reversao=20.0  # Boost para confirmação
                    )
        
        # CASO 3: Sem S/R próximo → CONFIRMAR ORIGINAL
        dados_extras['decisao_ia'] = 'SEM_SR_PROXIMO'
        return FiltroResult(
            bloqueado=False,
            motivo="",
            confianca=0.5,
            dados_extras=dados_extras
        )

class FiltroAntiArmadilha:
    """🪤 FILTRO ANTI-ARMADILHA: Detecta rompimentos falsos"""
    
    @staticmethod
    def detectar_armadilha(df: pd.DataFrame, tipo_sinal: str) -> FiltroResult:
        """Detecta possíveis armadilhas de rompimento"""
        if len(df) < 10:
            return FiltroResult(False, "", 0.0, {})
        
        closes = df['close'].values[-10:]
        volumes = df['volume'].values[-10:]
        highs = df['high'].values[-10:]
        lows = df['low'].values[-10:]
        
        # 1. ROMPIMENTO RECENTE
        movimento_forte = False
        direcao_movimento = None
        
        # Verificar movimento nas últimas 5 velas
        if len(closes) >= 5:
            variacao_5v = (closes[-1] - closes[-5]) / closes[-5]
            if abs(variacao_5v) > 0.015:  # Movimento > 1.5%
                movimento_forte = True
                direcao_movimento = 'UP' if variacao_5v > 0 else 'DOWN'
        
        # 2. VOLUME NO ROMPIMENTO
        volume_confirmacao = False
        if len(volumes) >= 5:
            vol_medio_anterior = np.mean(volumes[-5:-1])
            vol_rompimento = volumes[-1]
            if vol_rompimento > vol_medio_anterior * 1.5:
                volume_confirmacao = True
        
        # 3. REJEIÇÃO IMEDIATA (Sombras longas)
        rejeicao_detectada = False
        if len(df) >= 2:
            # Vela atual
            vela_atual = df.iloc[-1]
            o1, h1, l1, c1 = vela_atual['open'], vela_atual['high'], vela_atual['low'], vela_atual['close']
            
            # Vela anterior
            vela_anterior = df.iloc[-2]
            o2, h2, l2, c2 = vela_anterior['open'], vela_anterior['high'], vela_anterior['low'], vela_anterior['close']
            
            range1 = h1 - l1
            range2 = h2 - l2
            
            if range1 > 0 and range2 > 0:
                # Sombra superior longa após movimento de alta
                sombra_sup1 = (h1 - max(o1, c1)) / range1
                if (direcao_movimento == 'UP' and sombra_sup1 > 0.5 and 
                    c1 < (h1 + l1) / 2):  # Fechou na parte inferior
                    rejeicao_detectada = True
                
                # Sombra inferior longa após movimento de baixa
                sombra_inf1 = (min(o1, c1) - l1) / range1
                if (direcao_movimento == 'DOWN' and sombra_inf1 > 0.5 and 
                    c1 > (h1 + l1) / 2):  # Fechou na parte superior
                    rejeicao_detectada = True
        
        # 4. DIVERGÊNCIA COM SINAL
        divergencia = False
        if movimento_forte and direcao_movimento:
            # CALL após movimento de baixa (possível pegadinha)
            if 'CALL' in tipo_sinal and direcao_movimento == 'DOWN':
                divergencia = True
            
            # PUT após movimento de alta (possível pegadinha)
            if 'PUT' in tipo_sinal and direcao_movimento == 'UP':
                divergencia = True
        
        # DECISÃO DE BLOQUEIO
        sinais_armadilha = 0
        motivos = []
        
        if movimento_forte and not volume_confirmacao:
            sinais_armadilha += 2
            motivos.append("Rompimento sem volume")
        
        if rejeicao_detectada:
            sinais_armadilha += 2
            motivos.append("Rejeição imediata detectada")
        
        if divergencia:
            sinais_armadilha += 1
            motivos.append(f"Sinal contra movimento ({direcao_movimento})")
        
        # Bloquear se 2+ sinais de armadilha
        bloqueado = sinais_armadilha >= 2
        confianca = min(sinais_armadilha / 4.0, 1.0)
        
        motivo = f"Armadilha detectada: {', '.join(motivos)}" if bloqueado else ""
        
        return FiltroResult(
            bloqueado=bloqueado,
            motivo=motivo,
            confianca=confianca,
            dados_extras={
                'movimento_forte': movimento_forte,
                'direcao_movimento': direcao_movimento,
                'volume_confirmacao': volume_confirmacao,
                'rejeicao_detectada': rejeicao_detectada,
                'divergencia': divergencia,
                'sinais_armadilha': sinais_armadilha
            }
        )

class FiltroAntiLateral:
    """📏 FILTRO ANTI-LATERAL: Detecta mercados lateralizados"""
    
    @staticmethod
    def detectar_lateralizacao(df: pd.DataFrame, periodo: int = 20) -> FiltroResult:
        """Detecta mercados em lateralização"""
        if len(df) < periodo:
            return FiltroResult(False, "", 0.0, {})
        
        closes = df['close'].values[-periodo:]
        highs = df['high'].values[-periodo:]
        lows = df['low'].values[-periodo:]
        
        # 1. RANGE DE PREÇOS
        max_price = np.max(highs)
        min_price = np.min(lows)
        range_pct = ((max_price - min_price) / min_price) * 100
        
        # 2. DESVIO PADRÃO DOS FECHAMENTOS
        std_closes = np.std(closes)
        mean_closes = np.mean(closes)
        cv = (std_closes / mean_closes) * 100  # Coeficiente de variação
        
        # 3. TENDÊNCIA LINEAR
        x = np.arange(len(closes))
        correlation = np.corrcoef(x, closes)[0, 1]
        tendencia_fraca = abs(correlation) < 0.3
        
        # 4. OSCILAÇÕES PEQUENAS
        movimentos = np.abs(np.diff(closes))
        movimento_medio = np.mean(movimentos)
        movimento_medio_pct = (movimento_medio / mean_closes) * 100
        
        # 5. ANÁLISE DE VELAS (Corpos pequenos)
        if len(df) >= 10:
            velas_recentes = df.iloc[-10:]
            corpos_pequenos = 0
            
            for _, vela in velas_recentes.iterrows():
                o, h, l, c = vela['open'], vela['high'], vela['low'], vela['close']
                corpo = abs(c - o)
                range_vela = h - l
                
                if range_vela > 0 and (corpo / range_vela) < 0.3:
                    corpos_pequenos += 1
            
            alta_indecisao = corpos_pequenos >= 6  # 60% das velas
        else:
            alta_indecisao = False
        
        # DECISÃO DE BLOQUEIO - 🚀 MAIS RIGOROSO
        sinais_lateral = 0
        motivos = []
        
        if range_pct < 0.08:  # Range muito pequeno (mais rigoroso)
            sinais_lateral += 2
            motivos.append(f"Range pequeno: {range_pct:.3f}%")
        
        if cv < 0.5:  # Baixa volatilidade (mais rigoroso)
            sinais_lateral += 1
            motivos.append(f"Baixa volatilidade: {cv:.2f}%")
        
        if tendencia_fraca:
            sinais_lateral += 1
            motivos.append(f"Sem tendência clara: r={correlation:.2f}")
        
        if movimento_medio_pct < 0.03:  # Movimentos muito pequenos (mais rigoroso)
            sinais_lateral += 1
            motivos.append(f"Movimentos pequenos: {movimento_medio_pct:.3f}%")
        
        if alta_indecisao:
            sinais_lateral += 1
            motivos.append(f"Indecisão: {corpos_pequenos}/10 velas pequenas")
        
        # Bloquear se 4+ sinais de lateralização (mais rigoroso)
        bloqueado = sinais_lateral >= 4
        confianca = min(sinais_lateral / 5.0, 1.0)
        
        motivo = f"Lateralização: {', '.join(motivos)}" if bloqueado else ""
        
        return FiltroResult(
            bloqueado=bloqueado,
            motivo=motivo,
            confianca=confianca,
            dados_extras={
                'range_pct': range_pct,
                'volatilidade': cv,
                'correlacao_tendencia': correlation,
                'movimento_medio_pct': movimento_medio_pct,
                'corpos_pequenos': corpos_pequenos if 'corpos_pequenos' in locals() else 0,
                'sinais_lateral': sinais_lateral
            }
        )

class FiltroAntiGap:
    """⚡ FILTRO ANTI-GAP: Detecta movimentos extremos anômalos"""
    
    @staticmethod
    def detectar_gap_anomalo(df: pd.DataFrame, tipo_sinal: str) -> FiltroResult:
        """Detecta gaps e movimentos extremos anômalos"""
        if len(df) < 5:
            return FiltroResult(False, "", 0.0, {})
        
        closes = df['close'].values[-5:]
        volumes = df['volume'].values[-5:]
        
        # 1. MOVIMENTO EXTREMO RECENTE
        movimento_1v = abs((closes[-1] - closes[-2]) / closes[-2]) * 100
        movimento_3v = abs((closes[-1] - closes[-3]) / closes[-3]) * 100
        
        # 2. VOLATILIDADE ANÔMALA
        volatilidade_recente = np.std(closes[-3:]) / np.mean(closes[-3:]) * 100
        volatilidade_anterior = np.std(closes[-5:-2]) / np.mean(closes[-5:-2]) * 100
        
        spike_volatilidade = volatilidade_recente > volatilidade_anterior * 3
        
        # 3. VOLUME ANÔMALO
        volume_atual = volumes[-1]
        volume_medio = np.mean(volumes[-4:-1])
        volume_extremo = volume_atual > volume_medio * 5
        
        # 4. HORÁRIO SUSPEITO (Baixa liquidez)
        hora_atual = datetime.datetime.now().hour
        horario_baixa_liquidez = (2 <= hora_atual <= 6)
        
        # 5. SEQUÊNCIA DE GAPS
        gaps_consecutivos = 0
        for i in range(1, len(closes)):
            movimento = abs((closes[i] - closes[i-1]) / closes[i-1]) * 100
            if movimento > 0.5:  # Movimento > 0.5%
                gaps_consecutivos += 1
        
        # DECISÃO DE BLOQUEIO
        sinais_gap = 0
        motivos = []
        
        if movimento_1v > 1.0:  # Movimento muito extremo
            sinais_gap += 2
            motivos.append(f"Movimento extremo: {movimento_1v:.2f}%")
        
        if spike_volatilidade:
            sinais_gap += 1
            motivos.append("Spike de volatilidade")
        
        if volume_extremo and horario_baixa_liquidez:
            sinais_gap += 2
            motivos.append("Volume anômalo + horário suspeito")
        
        if gaps_consecutivos >= 3:
            sinais_gap += 1
            motivos.append(f"{gaps_consecutivos} movimentos extremos")
        
        # Bloquear se 2+ sinais de gap/anomalia
        bloqueado = sinais_gap >= 2
        confianca = min(sinais_gap / 4.0, 1.0)
        
        motivo = f"Gap/Anomalia: {', '.join(motivos)}" if bloqueado else ""
        
        return FiltroResult(
            bloqueado=bloqueado,
            motivo=motivo,
            confianca=confianca,
            dados_extras={
                'movimento_1v': movimento_1v,
                'movimento_3v': movimento_3v,
                'spike_volatilidade': spike_volatilidade,
                'volume_extremo': volume_extremo,
                'horario_baixa_liquidez': horario_baixa_liquidez,
                'gaps_consecutivos': gaps_consecutivos,
                'sinais_gap': sinais_gap
            }
        )

class FiltrosContextoMaster:
    """🛡️ MASTER CLASS: Coordena todos os filtros contextuais + IA SUPERVISORA"""
    
    def __init__(self):
        self.filtro_exaustao = FiltroAntiExaustao()
        self.filtro_sr = FiltroSupporteResistenciaReal()
        self.filtro_armadilha = FiltroAntiArmadilha()
        self.filtro_lateral = FiltroAntiLateral()
        self.filtro_gap = FiltroAntiGap()
        
        self.stats = {
            'total_analises': 0,
            'bloqueios_exaustao': 0,
            'bloqueios_sr': 0,
            'bloqueios_armadilha': 0,
            'bloqueios_lateral': 0,
            'bloqueios_gap': 0,
            'bloqueios_multiplos': 0,
            'reversoes_sugeridas': 0,  # 🚀 NOVO
            'confirmacoes_ia': 0       # 🚀 NOVO
        }
        
        # Configurações
        self.config = {
            'usar_filtro_exaustao': True,
            'usar_filtro_sr': True,
            'usar_filtro_armadilha': True,
            'usar_filtro_lateral': True,
            'usar_filtro_gap': True,
            'score_minimo_confianca': 0.6,
            'permitir_filtros_multiplos': True,
            'ia_supervisora_ativa': True  # 🚀 NOVO
        }
    
    def analisar_contexto_completo(self, df: pd.DataFrame, analise_completa: Dict, 
                                  par: str, tipo_sinal: str, score_total: float) -> ContextAnalysis:
        """🚀 ANÁLISE COMPLETA COM IA SUPERVISORA INTELIGENTE"""
        
        self.stats['total_analises'] += 1
        
        filtros_aplicados = []
        motivos_bloqueio = []
        scores_confianca = []
        ajuste_score_total = 0
        dados_contexto = {}
        entrada_segura = True
        
        # 🚀 VARIÁVEIS IA SUPERVISORA
        ia_sugestao = None
        tipo_sugerido_ia = None
        motivo_ia = ""
        score_boost_ia = 0.0
        
        # RSI para filtro de exaustão
        rsi = analise_completa.get('rsi', 50)
        
        # 1. FILTRO ANTI-EXAUSTÃO
        if self.config['usar_filtro_exaustao']:
            resultado_exaustao = self.filtro_exaustao.detectar_exaustao(df, tipo_sinal, rsi)
            filtros_aplicados.append('anti_exaustao')
            dados_contexto['exaustao'] = resultado_exaustao.dados_extras
            
            if resultado_exaustao.sugerir_reversao:
                ia_sugestao = 'REVERTER'
                tipo_sugerido_ia = resultado_exaustao.tipo_sugerido
                motivo_ia = f"Exaustão detectada - Sugerindo {resultado_exaustao.tipo_sugerido}"
                score_boost_ia = resultado_exaustao.score_boost_reversao
                self.stats['reversoes_sugeridas'] += 1
            elif resultado_exaustao.bloqueado:
                entrada_segura = False
                motivos_bloqueio.append(resultado_exaustao.motivo)
                self.stats['bloqueios_exaustao'] += 1
                scores_confianca.append(resultado_exaustao.confianca)
        
        # 2. 🚀 FILTRO S/R REAL COM IA SUPERVISORA (PRINCIPAL CORREÇÃO)
        if self.config['usar_filtro_sr']:
            resultado_sr = self.filtro_sr.analisar_sr_real(df, tipo_sinal)
            filtros_aplicados.append('sr_real')
            dados_contexto['sr'] = resultado_sr.dados_extras
            
            # 🧠 IA SUPERVISORA EM AÇÃO
            if resultado_sr.sugerir_reversao:
                ia_sugestao = 'REVERTER'
                tipo_sugerido_ia = resultado_sr.tipo_sugerido
                
                # Mensagens específicas baseadas na decisão da IA
                if 'decisao_ia' in resultado_sr.dados_extras:
                    decisao = resultado_sr.dados_extras['decisao_ia']
                    if decisao == 'REVERSAO_RESISTENCIA':
                        nivel = resultado_sr.dados_extras['nivel_conflitante']['nivel']
                        toques = resultado_sr.dados_extras['nivel_conflitante']['toques']
                        motivo_ia = f"🧠 IA: CALL em resistência {nivel:.5f} ({toques} toques) → Sugerindo PUT"
                    elif decisao == 'REVERSAO_SUPORTE':
                        nivel = resultado_sr.dados_extras['nivel_conflitante']['nivel']
                        toques = resultado_sr.dados_extras['nivel_conflitante']['toques']
                        motivo_ia = f"🧠 IA: PUT em suporte {nivel:.5f} ({toques} toques) → Sugerindo CALL"
                    elif decisao == 'CONFIRMACAO_SUPORTE':
                        nivel = resultado_sr.dados_extras['nivel_favoravel']['nivel']
                        motivo_ia = f"🧠 IA: CALL próximo suporte {nivel:.5f} → Confirmando"
                    elif decisao == 'CONFIRMACAO_RESISTENCIA':
                        nivel = resultado_sr.dados_extras['nivel_favoravel']['nivel']
                        motivo_ia = f"🧠 IA: PUT próximo resistência {nivel:.5f} → Confirmando"
                    else:
                        motivo_ia = f"🧠 IA: Zona S/R detectada → Sugerindo {resultado_sr.tipo_sugerido}"
                
                score_boost_ia = resultado_sr.score_boost_reversao
                self.stats['reversoes_sugeridas'] += 1
                
                # 🚀 IMPORTANTE: NÃO BLOQUEIA, APENAS SUGERE REVERSÃO
                print(f"🧠 IA SUPERVISORA: {par.upper()} {motivo_ia}")
                
            elif resultado_sr.score_boost_reversao > 0:
                # IA confirmou a estratégia original
                ia_sugestao = 'CONFIRMAR'
                score_boost_ia = resultado_sr.score_boost_reversao
                motivo_ia = "🧠 IA: S/R favorável à estratégia original"
                self.stats['confirmacoes_ia'] += 1
            
            # Só bloqueia se for um bloqueio real (não deveria acontecer mais)
            if resultado_sr.bloqueado:
                entrada_segura = False
                motivos_bloqueio.append(resultado_sr.motivo)
                self.stats['bloqueios_sr'] += 1
                scores_confianca.append(resultado_sr.confianca)
        
        # 3. FILTRO ANTI-ARMADILHA
        if self.config['usar_filtro_armadilha']:
            resultado_armadilha = self.filtro_armadilha.detectar_armadilha(df, tipo_sinal)
            filtros_aplicados.append('anti_armadilha')
            dados_contexto['armadilha'] = resultado_armadilha.dados_extras
            
            if resultado_armadilha.bloqueado:
                entrada_segura = False
                motivos_bloqueio.append(resultado_armadilha.motivo)
                self.stats['bloqueios_armadilha'] += 1
                scores_confianca.append(resultado_armadilha.confianca)
        
        # 4. FILTRO ANTI-LATERAL
        if self.config['usar_filtro_lateral']:
            resultado_lateral = self.filtro_lateral.detectar_lateralizacao(df)
            filtros_aplicados.append('anti_lateral')
            dados_contexto['lateral'] = resultado_lateral.dados_extras
            
            if resultado_lateral.bloqueado:
                entrada_segura = False
                motivos_bloqueio.append(resultado_lateral.motivo)
                self.stats['bloqueios_lateral'] += 1
                scores_confianca.append(resultado_lateral.confianca)
        
        # 5. FILTRO ANTI-GAP
        if self.config['usar_filtro_gap']:
            resultado_gap = self.filtro_gap.detectar_gap_anomalo(df, tipo_sinal)
            filtros_aplicados.append('anti_gap')
            dados_contexto['gap'] = resultado_gap.dados_extras
            
            if resultado_gap.bloqueado:
                entrada_segura = False
                motivos_bloqueio.append(resultado_gap.motivo)
                self.stats['bloqueios_gap'] += 1
                scores_confianca.append(resultado_gap.confianca)
        
        # ANÁLISE CONSOLIDADA
        if len(motivos_bloqueio) > 1:
            self.stats['bloqueios_multiplos'] += 1
        
        # Score de confiança médio
        score_confianca = np.mean(scores_confianca) if scores_confianca else 0.0
        
        # 🚀 AJUSTE DE SCORE BASEADO NA IA SUPERVISORA
        if ia_sugestao == 'REVERTER':
            # IA sugeriu reversão - aplicar boost na direção sugerida
            ajuste_score_total = score_boost_ia
        elif ia_sugestao == 'CONFIRMAR':
            # IA confirmou estratégia original - pequeno boost
            ajuste_score_total = score_boost_ia
        elif not entrada_segura:
            # Bloqueios reais (casos extremos)
            ajuste_score_total = -min(50, score_confianca * 100)
        else:
            # Passou por todos os filtros sem problemas
            ajuste_score_total = 5
        
        # Log apenas para bloqueios reais (não para reversões da IA)
        if motivos_bloqueio:
            print(f"🛡️ FILTROS CONTEXTO: {par.upper()} {tipo_sinal} BLOQUEADO")
            for motivo in motivos_bloqueio:
                print(f"   🚫 {motivo}")
        
        return ContextAnalysis(
            entrada_segura=entrada_segura,
            filtros_aplicados=filtros_aplicados,
            motivos_bloqueio=motivos_bloqueio,
            score_confianca=score_confianca,
            ajuste_score=ajuste_score_total,
            dados_contexto=dados_contexto,
            ia_sugestao=ia_sugestao,
            tipo_sugerido_ia=tipo_sugerido_ia,
            motivo_ia=motivo_ia,
            score_boost_ia=score_boost_ia
        )
    
    def configurar_filtros(self, **kwargs):
        """Configura quais filtros usar"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"🛡️ Filtro configurado: {key} = {value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos filtros"""
        total = self.stats['total_analises']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'taxa_bloqueio_total': ((sum(self.stats[k] for k in self.stats if 'bloqueios' in k and k != 'bloqueios_multiplos') / total * 100) if total > 0 else 0),
            'taxa_bloqueio_exaustao': (self.stats['bloqueios_exaustao'] / total * 100),
            'taxa_bloqueio_sr': (self.stats['bloqueios_sr'] / total * 100),
            'taxa_bloqueio_armadilha': (self.stats['bloqueios_armadilha'] / total * 100),
            'taxa_bloqueio_lateral': (self.stats['bloqueios_lateral'] / total * 100),
            'taxa_bloqueio_gap': (self.stats['bloqueios_gap'] / total * 100),
            'taxa_reversoes_ia': (self.stats['reversoes_sugeridas'] / total * 100),
            'taxa_confirmacoes_ia': (self.stats['confirmacoes_ia'] / total * 100),
            'filtros_ativos': sum(1 for k, v in self.config.items() if k.startswith('usar_') and v)
        }
    
    def gerar_relatorio_filtros(self) -> str:
        """Gera relatório detalhado dos filtros"""
        stats = self.get_stats()
        
        relatorio = f"""
🛡️ RELATÓRIO FILTROS CONTEXTO + IA SUPERVISORA

📊 ESTATÍSTICAS GERAIS:
   Total Análises: {stats['total_analises']}
   Taxa Bloqueio: {stats.get('taxa_bloqueio_total', 0):.1f}%
   Filtros Ativos: {stats.get('filtros_ativos', 0)}/5

🧠 IA SUPERVISORA:
   🔄 Reversões Sugeridas: {stats['reversoes_sugeridas']} ({stats.get('taxa_reversoes_ia', 0):.1f}%)
   ✅ Confirmações IA: {stats['confirmacoes_ia']} ({stats.get('taxa_confirmacoes_ia', 0):.1f}%)

🚫 BLOQUEIOS POR FILTRO:
   🔴 Anti-Exaustão: {stats['bloqueios_exaustao']} ({stats.get('taxa_bloqueio_exaustao', 0):.1f}%)
   📊 S/R Real: {stats['bloqueios_sr']} ({stats.get('taxa_bloqueio_sr', 0):.1f}%)
   🪤 Anti-Armadilha: {stats['bloqueios_armadilha']} ({stats.get('taxa_bloqueio_armadilha', 0):.1f}%)
   📏 Anti-Lateral: {stats['bloqueios_lateral']} ({stats.get('taxa_bloqueio_lateral', 0):.1f}%)
   ⚡ Anti-Gap: {stats['bloqueios_gap']} ({stats.get('taxa_bloqueio_gap', 0):.1f}%)
   🎯 Múltiplos: {stats['bloqueios_multiplos']}

⚙️ CONFIGURAÇÃO ATUAL:
"""
        
        for key, value in self.config.items():
            if key.startswith('usar_'):
                filtro_nome = key.replace('usar_filtro_', '').replace('_', ' ').title()
                status = "✅ ATIVO" if value else "❌ INATIVO"
                relatorio += f"   {filtro_nome}: {status}\n"
        
        relatorio += f"   🧠 IA Supervisora: {'✅ ATIVO' if self.config.get('ia_supervisora_ativa', True) else '❌ INATIVO'}\n"
        relatorio += "\n🧠 IA SUPERVISORA TRABALHANDO EM HARMONIA COM INDICADORES!"
        
        return relatorio

# Classe de integração para o sistema principal
class FiltrosIntegration:
    """Classe de integração para o sistema principal"""
    
    def __init__(self):
        self.filtros_master = FiltrosContextoMaster()
        print("🛡️ Filtros Contexto + IA Supervisora inicializados!")
    
    def validar_contexto(self, df: pd.DataFrame, analise_completa: Dict, 
                        par: str, tipo_sinal: str, score_total: float) -> Dict[str, Any]:
        """🚀 MÉTODO PRINCIPAL CORRIGIDO - COM IA SUPERVISORA"""
        
        resultado = self.filtros_master.analisar_contexto_completo(
            df, analise_completa, par, tipo_sinal, score_total
        )
        
        return {
            'entrada_segura': resultado.entrada_segura,
            'filtros_aplicados': resultado.filtros_aplicados,
            'motivos_bloqueio': resultado.motivos_bloqueio,
            'score_confianca': resultado.score_confianca,
            'ajuste_score': resultado.ajuste_score,
            'dados_contexto': resultado.dados_contexto,
            'total_filtros': len(resultado.filtros_aplicados),
            'filtros_bloquearam': len(resultado.motivos_bloqueio) > 0,
            # 🚀 NOVOS CAMPOS IA SUPERVISORA
            'ia_sugestao': resultado.ia_sugestao,
            'tipo_sugerido_ia': resultado.tipo_sugerido_ia,
            'motivo_ia': resultado.motivo_ia,
            'score_boost_ia': resultado.score_boost_ia,
            'ia_supervisora_ativa': True
        }
    
    def configurar(self, **kwargs):
        """Configura os filtros"""
        self.filtros_master.configurar_filtros(**kwargs)
    
    def get_stats(self):
        """Retorna estatísticas"""
        return self.filtros_master.get_stats()
    
    def gerar_relatorio(self):
        """Gera relatório"""
        return self.filtros_master.gerar_relatorio_filtros()

print("✅ FILTROS CONTEXTO + IA SUPERVISORA CARREGADOS - HARMONY ACTIVE!")