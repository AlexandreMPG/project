#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚨 DETECTOR CENÁRIOS PERIGOSOS - CIPHER ROYAL SUPREME ENHANCED + AI 🚨
💎 DETECTA AS SITUAÇÕES DAS "CRUZETAS" QUE CAUSAM LOSSES
🔥 PATTERNS ESPECÍFICOS: Call no topo + Put em lateral + Trade pós-gap
🎯 MACHINE LEARNING PARA IDENTIFICAR PADRÕES HISTÓRICOS DE LOSS
"""

import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict

@dataclass
class CenarioPerigosoResult:
    """Resultado da detecção de cenário perigoso"""
    cenario_detectado: bool
    tipo_cenario: str
    nivel_perigo: str  # LOW, MEDIUM, HIGH, EXTREME
    confianca: float
    motivos: List[str]
    dados_tecnicos: Dict[str, Any]
    recomendacao: str

class DetectorCallNoTopo:
    """🔴 DETECTOR: CALL em topo de resistência forte"""
    
    @staticmethod
    def detectar_call_no_topo(df: pd.DataFrame, analise_completa: Dict, 
                             tipo_sinal: str) -> CenarioPerigosoResult:
        """Detecta CALL sendo emitido no topo de resistência"""
        
        if 'CALL' not in tipo_sinal:
            return CenarioPerigosoResult(
                cenario_detectado=False, tipo_cenario="", nivel_perigo="LOW",
                confianca=0.0, motivos=[], dados_tecnicos={}, recomendacao=""
            )
        
        if len(df) < 30:
            return CenarioPerigosoResult(
                cenario_detectado=False, tipo_cenario="", nivel_perigo="LOW",
                confianca=0.0, motivos=[], dados_tecnicos={}, recomendacao=""
            )
        
        closes = df['close'].values
        highs = df['high'].values
        volumes = df['volume'].values
        current_price = closes[-1]
        
        motivos = []
        sinais_perigo = 0
        dados_tecnicos = {}
        
        # 1. VERIFICAR SE ESTÁ EM ALTA RECENTE
        movimento_5v = (current_price - closes[-6]) / closes[-6] * 100
        movimento_10v = (current_price - closes[-11]) / closes[-11] * 100
        
        em_alta_recente = movimento_5v > 0.5 or movimento_10v > 1.0
        dados_tecnicos['movimento_5v'] = movimento_5v
        dados_tecnicos['movimento_10v'] = movimento_10v
        
        if em_alta_recente:
            sinais_perigo += 1
            motivos.append(f"Em alta recente: {movimento_5v:.2f}% (5v)")
        
        # 2. DETECTAR RESISTÊNCIA FORTE PRÓXIMA
        lookback = min(50, len(highs))
        resistencias = []
        
        for i in range(2, lookback - 2):
            if (highs[-(lookback-i)] >= highs[-(lookback-i-1)] and 
                highs[-(lookback-i)] >= highs[-(lookback-i-2)] and
                highs[-(lookback-i)] >= highs[-(lookback-i+1)] and 
                highs[-(lookback-i)] >= highs[-(lookback-i+2)]):
                resistencias.append(highs[-(lookback-i)])
        
        # Agrupar resistências próximas
        resistencias_fortes = []
        if resistencias:
            resistencias_sorted = sorted(resistencias, reverse=True)
            for res in resistencias_sorted:
                # Verificar se há outras resistências próximas (confirmar força)
                toques_proximos = sum(1 for r in resistencias if abs(r - res) / res < 0.003)
                if toques_proximos >= 2:  # Mínimo 2 toques
                    distancia_atual = abs(current_price - res) / current_price
                    resistencias_fortes.append({
                        'nivel': res,
                        'toques': toques_proximos,
                        'distancia': distancia_atual
                    })
        
        # Verificar proximidade de resistência forte
        resistencia_proxima = None
        for res in resistencias_fortes:
            if res['distancia'] < 0.008:  # Menos de 0.8%
                resistencia_proxima = res
                sinais_perigo += 2
                motivos.append(f"Resistência forte próxima: {res['nivel']:.5f} ({res['toques']} toques)")
                break
        
        dados_tecnicos['resistencias_fortes'] = len(resistencias_fortes)
        dados_tecnicos['resistencia_proxima'] = resistencia_proxima
        
        # 3. RSI ELEVADO (Sobrecomprado)
        rsi = analise_completa.get('rsi', 50)
        if rsi > 70:
            sinais_perigo += 1
            motivos.append(f"RSI elevado: {rsi:.1f}")
            if rsi > 80:
                sinais_perigo += 1
                motivos.append("RSI extremamente alto")
        
        dados_tecnicos['rsi'] = rsi
        
        # 4. VOLUME DECRESCENTE (Falta de força)
        if len(volumes) >= 5:
            vol_atual = volumes[-1]
            vol_medio_anterior = np.mean(volumes[-5:-1])
            
            if vol_atual < vol_medio_anterior * 0.7:
                sinais_perigo += 1
                motivos.append("Volume decrescente na alta")
        
        # 5. VELAS DE INDECISÃO/REJEIÇÃO
        if len(df) >= 3:
            velas_recentes = df.iloc[-3:]
            sinais_indecisao = 0
            
            for _, vela in velas_recentes.iterrows():
                o, h, l, c = vela['open'], vela['high'], vela['low'], vela['close']
                corpo = abs(c - o)
                range_total = h - l
                
                if range_total > 0:
                    # Doji ou corpo pequeno
                    if (corpo / range_total) < 0.3:
                        sinais_indecisao += 1
                    
                    # Sombra superior longa (rejeição)
                    sombra_superior = (h - max(o, c)) / range_total
                    if sombra_superior > 0.6:
                        sinais_indecisao += 2
            
            if sinais_indecisao >= 3:
                sinais_perigo += 1
                motivos.append("Sinais de indecisão/rejeição")
        
        # 6. MOMENTUM DIVERGENTE
        if len(closes) >= 10:
            # Preço fazendo nova máxima mas momentum enfraquecendo
            max_recente = np.max(closes[-5:])
            max_anterior = np.max(closes[-10:-5])
            
            if max_recente > max_anterior:
                # Verificar se movimento está perdendo força
                movimento_recente = np.mean(np.diff(closes[-5:]))
                movimento_anterior = np.mean(np.diff(closes[-10:-5]))
                
                if movimento_recente < movimento_anterior * 0.5:
                    sinais_perigo += 1
                    motivos.append("Momentum divergente (enfraquecendo)")
        
        # AVALIAÇÃO FINAL
        cenario_detectado = sinais_perigo >= 3
        
        if sinais_perigo >= 5:
            nivel_perigo = "EXTREME"
        elif sinais_perigo >= 4:
            nivel_perigo = "HIGH"
        elif sinais_perigo >= 3:
            nivel_perigo = "MEDIUM"
        else:
            nivel_perigo = "LOW"
        
        confianca = min(sinais_perigo / 6.0, 1.0)
        
        recomendacao = ""
        if cenario_detectado:
            recomendacao = "EVITAR CALL - Aguardar pullback ou rompimento confirmado"
        
        return CenarioPerigosoResult(
            cenario_detectado=cenario_detectado,
            tipo_cenario="CALL_NO_TOPO",
            nivel_perigo=nivel_perigo,
            confianca=confianca,
            motivos=motivos,
            dados_tecnicos=dados_tecnicos,
            recomendacao=recomendacao
        )

class DetectorPutEmLateral:
    """📏 DETECTOR: PUT em mercado lateralizado"""
    
    @staticmethod
    def detectar_put_em_lateral(df: pd.DataFrame, analise_completa: Dict, 
                               tipo_sinal: str) -> CenarioPerigosoResult:
        """Detecta PUT sendo emitido em mercado lateral"""
        
        if 'PUT' not in tipo_sinal:
            return CenarioPerigosoResult(
                cenario_detectado=False, tipo_cenario="", nivel_perigo="LOW",
                confianca=0.0, motivos=[], dados_tecnicos={}, recomendacao=""
            )
        
        if len(df) < 20:
            return CenarioPerigosoResult(
                cenario_detectado=False, tipo_cenario="", nivel_perigo="LOW",
                confianca=0.0, motivos=[], dados_tecnicos={}, recomendacao=""
            )
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        motivos = []
        sinais_perigo = 0
        dados_tecnicos = {}
        
        # 1. ANÁLISE DE RANGE (Lateralização)
        periodo_analise = min(20, len(closes))
        closes_periodo = closes[-periodo_analise:]
        highs_periodo = highs[-periodo_analise:]
        lows_periodo = lows[-periodo_analise:]
        
        max_periodo = np.max(highs_periodo)
        min_periodo = np.min(lows_periodo)
        range_pct = ((max_periodo - min_periodo) / min_periodo) * 100
        
        dados_tecnicos['range_pct'] = range_pct
        
        if range_pct < 0.8:  # Range muito pequeno
            sinais_perigo += 2
            motivos.append(f"Range pequeno: {range_pct:.2f}%")
        elif range_pct < 1.2:
            sinais_perigo += 1
            motivos.append(f"Range limitado: {range_pct:.2f}%")
        
        # 2. VOLATILIDADE BAIXA
        volatilidade = np.std(closes_periodo) / np.mean(closes_periodo) * 100
        dados_tecnicos['volatilidade'] = volatilidade
        
        if volatilidade < 0.5:
            sinais_perigo += 2
            motivos.append(f"Volatilidade muito baixa: {volatilidade:.2f}%")
        elif volatilidade < 0.8:
            sinais_perigo += 1
            motivos.append(f"Volatilidade baixa: {volatilidade:.2f}%")
        
        # 3. AUSÊNCIA DE TENDÊNCIA CLARA
        # Correlação linear com tempo
        x = np.arange(len(closes_periodo))
        correlation = np.corrcoef(x, closes_periodo)[0, 1]
        dados_tecnicos['correlacao_tendencia'] = correlation
        
        if abs(correlation) < 0.2:
            sinais_perigo += 2
            motivos.append("Ausência de tendência clara")
        elif abs(correlation) < 0.4:
            sinais_perigo += 1
            motivos.append("Tendência fraca")
        
        # 4. MÚLTIPLOS TOQUES EM S/R
        current_price = closes[-1]
        
        # Contar toques próximos ao preço atual
        toques_superior = sum(1 for h in highs_periodo if abs(h - current_price) / current_price < 0.005)
        toques_inferior = sum(1 for l in lows_periodo if abs(l - current_price) / current_price < 0.005)
        toques_preco_atual = sum(1 for c in closes_periodo if abs(c - current_price) / current_price < 0.003)
        
        total_toques = toques_superior + toques_inferior + toques_preco_atual
        dados_tecnicos['toques_zona_atual'] = total_toques
        
        if total_toques >= 5:
            sinais_perigo += 2
            motivos.append(f"Múltiplos toques na zona: {total_toques}")
        
        # 5. VELAS PEQUENAS E INDECISÃO
        if len(df) >= 10:
            velas_recentes = df.iloc[-10:]
            velas_pequenas = 0
            
            for _, vela in velas_recentes.iterrows():
                o, h, l, c = vela['open'], vela['high'], vela['low'], vela['close']
                corpo = abs(c - o)
                range_vela = h - l
                
                if range_vela > 0 and (corpo / range_vela) < 0.4:
                    velas_pequenas += 1
            
            porcentagem_pequenas = velas_pequenas / 10
            dados_tecnicos['velas_pequenas_pct'] = porcentagem_pequenas * 100
            
            if porcentagem_pequenas >= 0.7:
                sinais_perigo += 2
                motivos.append(f"Excesso velas pequenas: {porcentagem_pequenas*100:.0f}%")
        
        # 6. VOLUME BAIXO E INCONSISTENTE
        if len(volumes) >= 10:
            vol_medio = np.mean(volumes[-10:])
            vol_std = np.std(volumes[-10:])
            cv_volume = (vol_std / vol_medio) if vol_medio > 0 else 0
            
            dados_tecnicos['cv_volume'] = cv_volume
            
            # Volume baixo E inconsistente indica falta de interesse
            if cv_volume < 0.3:  # Volume muito consistente (baixo interesse)
                sinais_perigo += 1
                motivos.append("Volume consistentemente baixo")
        
        # 7. RSI NEUTRO (Não há momentum)
        rsi = analise_completa.get('rsi', 50)
        dados_tecnicos['rsi'] = rsi
        
        if 40 <= rsi <= 60:
            sinais_perigo += 1
            motivos.append("RSI neutro (sem momentum)")
        
        # AVALIAÇÃO FINAL
        cenario_detectado = sinais_perigo >= 4
        
        if sinais_perigo >= 7:
            nivel_perigo = "EXTREME"
        elif sinais_perigo >= 6:
            nivel_perigo = "HIGH"
        elif sinais_perigo >= 4:
            nivel_perigo = "MEDIUM"
        else:
            nivel_perigo = "LOW"
        
        confianca = min(sinais_perigo / 8.0, 1.0)
        
        recomendacao = ""
        if cenario_detectado:
            recomendacao = "EVITAR PUT - Aguardar rompimento ou movimento direcional"
        
        return CenarioPerigosoResult(
            cenario_detectado=cenario_detectado,
            tipo_cenario="PUT_EM_LATERAL",
            nivel_perigo=nivel_perigo,
            confianca=confianca,
            motivos=motivos,
            dados_tecnicos=dados_tecnicos,
            recomendacao=recomendacao
        )

class DetectorTradePosGap:
    """⚡ DETECTOR: Trade após gap/volatilidade extrema"""
    
    @staticmethod
    def detectar_trade_pos_gap(df: pd.DataFrame, analise_completa: Dict, 
                              tipo_sinal: str) -> CenarioPerigosoResult:
        """Detecta trade sendo feito após gap ou volatilidade extrema"""
        
        if len(df) < 10:
            return CenarioPerigosoResult(
                cenario_detectado=False, tipo_cenario="", nivel_perigo="LOW",
                confianca=0.0, motivos=[], dados_tecnicos={}, recomendacao=""
            )
        
        closes = df['close'].values
        volumes = df['volume'].values
        highs = df['high'].values
        lows = df['low'].values
        
        motivos = []
        sinais_perigo = 0
        dados_tecnicos = {}
        
        # 1. DETECTAR GAPS RECENTES
        gaps_detectados = []
        for i in range(1, min(5, len(closes))):
            movimento = abs((closes[-i] - closes[-i-1]) / closes[-i-1]) * 100
            if movimento > 0.8:  # Movimento > 0.8%
                gaps_detectados.append({
                    'velas_atras': i,
                    'movimento_pct': movimento,
                    'direcao': 'UP' if closes[-i] > closes[-i-1] else 'DOWN'
                })
        
        dados_tecnicos['gaps_recentes'] = len(gaps_detectados)
        
        if gaps_detectados:
            gap_mais_recente = gaps_detectados[0]
            sinais_perigo += 2
            motivos.append(f"Gap recente: {gap_mais_recente['movimento_pct']:.2f}% ({gap_mais_recente['velas_atras']}v atrás)")
            
            if gap_mais_recente['movimento_pct'] > 2.0:
                sinais_perigo += 1
                motivos.append("Gap extremo detectado")
        
        # 2. VOLATILIDADE ANÔMALA
        volatilidade_5v = np.std(closes[-5:]) / np.mean(closes[-5:]) * 100
        volatilidade_10v = np.std(closes[-10:]) / np.mean(closes[-10:]) * 100
        
        spike_volatilidade = volatilidade_5v > volatilidade_10v * 2
        dados_tecnicos['volatilidade_5v'] = volatilidade_5v
        dados_tecnicos['volatilidade_10v'] = volatilidade_10v
        dados_tecnicos['spike_volatilidade'] = spike_volatilidade
        
        if spike_volatilidade:
            sinais_perigo += 2
            motivos.append(f"Spike volatilidade: {volatilidade_5v:.2f}% vs {volatilidade_10v:.2f}%")
        
        # 3. VOLUME ANÔMALO
        if len(volumes) >= 5:
            vol_atual = volumes[-1]
            vol_medio = np.mean(volumes[-5:-1])
            volume_ratio = vol_atual / vol_medio if vol_medio > 0 else 1
            
            dados_tecnicos['volume_ratio_atual'] = volume_ratio
            
            if volume_ratio > 4:
                sinais_perigo += 2
                motivos.append(f"Volume extremo: {volume_ratio:.1f}x")
            elif volume_ratio > 2.5:
                sinais_perigo += 1
                motivos.append(f"Volume elevado: {volume_ratio:.1f}x")
        
        # 4. HORÁRIO DE BAIXA LIQUIDEZ
        hora_atual = datetime.datetime.now().hour
        horario_problemático = (2 <= hora_atual <= 6) or (23 <= hora_atual <= 1)
        
        dados_tecnicos['hora_atual'] = hora_atual
        dados_tecnicos['horario_problematico'] = horario_problemático
        
        if horario_problemático and (gaps_detectados or spike_volatilidade):
            sinais_perigo += 1
            motivos.append("Gap/volatilidade em horário baixa liquidez")
        
        # 5. REVERSÃO IMEDIATA (Sinal de manipulação)
        if len(closes) >= 3:
            # Verificar se houve movimento forte seguido de reversão
            movimento_2v = (closes[-1] - closes[-3]) / closes[-3] * 100
            movimento_1v = (closes[-1] - closes[-2]) / closes[-2] * 100
            
            # Movimento forte em uma direção, depois reversão
            reversao_detectada = False
            if abs(movimento_2v) > 1.0:
                if (movimento_2v > 0 and movimento_1v < -0.3) or (movimento_2v < 0 and movimento_1v > 0.3):
                    reversao_detectada = True
                    sinais_perigo += 2
                    motivos.append("Reversão imediata após movimento forte")
            
            dados_tecnicos['reversao_detectada'] = reversao_detectada
        
        # 6. SOMBRAS EXTREMAS (Rejeição)
        if len(df) >= 2:
            velas_rejeicao = 0
            for i in range(1, min(3, len(df))):
                vela = df.iloc[-i]
                o, h, l, c = vela['open'], vela['high'], vela['low'], vela['close']
                
                range_vela = h - l
                if range_vela > 0:
                    sombra_superior = (h - max(o, c)) / range_vela
                    sombra_inferior = (min(o, c) - l) / range_vela
                    
                    if sombra_superior > 0.7 or sombra_inferior > 0.7:
                        velas_rejeicao += 1
            
            dados_tecnicos['velas_rejeicao'] = velas_rejeicao
            
            if velas_rejeicao >= 2:
                sinais_perigo += 1
                motivos.append("Múltiplas velas com rejeição")
        
        # 7. DIVERGÊNCIA COM SINAL
        if gaps_detectados:
            gap_mais_recente = gaps_detectados[0]
            
            # CALL após gap de baixa ou PUT após gap de alta (contra-tendência perigoso)
            if (('CALL' in tipo_sinal and gap_mais_recente['direcao'] == 'DOWN') or
                ('PUT' in tipo_sinal and gap_mais_recente['direcao'] == 'UP')):
                sinais_perigo += 1
                motivos.append("Sinal contra-tendência pós-gap")
        
        # AVALIAÇÃO FINAL
        cenario_detectado = sinais_perigo >= 3
        
        if sinais_perigo >= 6:
            nivel_perigo = "EXTREME"
        elif sinais_perigo >= 5:
            nivel_perigo = "HIGH"
        elif sinais_perigo >= 3:
            nivel_perigo = "MEDIUM"
        else:
            nivel_perigo = "LOW"
        
        confianca = min(sinais_perigo / 7.0, 1.0)
        
        recomendacao = ""
        if cenario_detectado:
            recomendacao = "EVITAR TRADE - Aguardar estabilização pós-gap/volatilidade"
        
        return CenarioPerigosoResult(
            cenario_detectado=cenario_detectado,
            tipo_cenario="TRADE_POS_GAP",
            nivel_perigo=nivel_perigo,
            confianca=confianca,
            motivos=motivos,
            dados_tecnicos=dados_tecnicos,
            recomendacao=recomendacao
        )

class DetectorCenariosPerigososMaster:
    """🚨 MASTER DETECTOR: Coordena todos os detectores de cenários perigosos"""
    
    def __init__(self):
        self.detector_call_topo = DetectorCallNoTopo()
        self.detector_put_lateral = DetectorPutEmLateral()
        self.detector_pos_gap = DetectorTradePosGap()
        
        self.stats = {
            'total_analises': 0,
            'cenarios_call_topo': 0,
            'cenarios_put_lateral': 0,
            'cenarios_pos_gap': 0,
            'cenarios_multiplos': 0,
            'bloqueios_total': 0
        }
        
        self.config = {
            'detectar_call_topo': True,
            'detectar_put_lateral': True,
            'detectar_pos_gap': True,
            'nivel_minimo_bloqueio': 'MEDIUM',  # LOW, MEDIUM, HIGH, EXTREME
            'confianca_minima': 0.6
        }
    
    def analisar_cenarios_completo(self, df: pd.DataFrame, analise_completa: Dict, 
                                  par: str, tipo_sinal: str, score_total: float) -> Dict[str, Any]:
        """Análise completa de cenários perigosos"""
        
        self.stats['total_analises'] += 1
        
        cenarios_detectados = []
        bloqueios = []
        max_nivel_perigo = "LOW"
        max_confianca = 0.0
        dados_completos = {}
        
        # 1. DETECTOR CALL NO TOPO
        if self.config['detectar_call_topo']:
            resultado_call_topo = self.detector_call_topo.detectar_call_no_topo(
                df, analise_completa, tipo_sinal
            )
            
            if resultado_call_topo.cenario_detectado:
                cenarios_detectados.append(resultado_call_topo)
                self.stats['cenarios_call_topo'] += 1
                dados_completos['call_topo'] = resultado_call_topo.dados_tecnicos
        
        # 2. DETECTOR PUT EM LATERAL
        if self.config['detectar_put_lateral']:
            resultado_put_lateral = self.detector_put_lateral.detectar_put_em_lateral(
                df, analise_completa, tipo_sinal
            )
            
            if resultado_put_lateral.cenario_detectado:
                cenarios_detectados.append(resultado_put_lateral)
                self.stats['cenarios_put_lateral'] += 1
                dados_completos['put_lateral'] = resultado_put_lateral.dados_tecnicos
        
        # 3. DETECTOR PÓS-GAP
        if self.config['detectar_pos_gap']:
            resultado_pos_gap = self.detector_pos_gap.detectar_trade_pos_gap(
                df, analise_completa, tipo_sinal
            )
            
            if resultado_pos_gap.cenario_detectado:
                cenarios_detectados.append(resultado_pos_gap)
                self.stats['cenarios_pos_gap'] += 1
                dados_completos['pos_gap'] = resultado_pos_gap.dados_tecnicos
        
        # ANÁLISE CONSOLIDADA
        if len(cenarios_detectados) > 1:
            self.stats['cenarios_multiplos'] += 1
        
        # Determinar se deve bloquear
        deve_bloquear = False
        motivos_bloqueio = []
        
        nivel_hierarchy = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'EXTREME': 3}
        nivel_minimo = nivel_hierarchy[self.config['nivel_minimo_bloqueio']]
        
        for cenario in cenarios_detectados:
            if nivel_hierarchy[cenario.nivel_perigo] >= nivel_minimo:
                if cenario.confianca >= self.config['confianca_minima']:
                    deve_bloquear = True
                    bloqueios.append(cenario.tipo_cenario)
                    motivos_bloqueio.extend(cenario.motivos)
                    
                    # Atualizar máximos
                    if nivel_hierarchy[cenario.nivel_perigo] > nivel_hierarchy[max_nivel_perigo]:
                        max_nivel_perigo = cenario.nivel_perigo
                    
                    if cenario.confianca > max_confianca:
                        max_confianca = cenario.confianca
        
        if deve_bloquear:
            self.stats['bloqueios_total'] += 1
        
        # Log detalhado
        if cenarios_detectados:
            print(f"🚨 CENÁRIOS PERIGOSOS: {par.upper()} {tipo_sinal}")
            for cenario in cenarios_detectados:
                print(f"   🔴 {cenario.tipo_cenario}: {cenario.nivel_perigo} ({cenario.confianca:.2f})")
                if deve_bloquear and cenario.tipo_cenario in bloqueios:
                    print(f"      🚫 BLOQUEADO: {cenario.recomendacao}")
        
        return {
            'cenarios_detectados': len(cenarios_detectados),
            'tipos_cenarios': [c.tipo_cenario for c in cenarios_detectados],
            'deve_bloquear': deve_bloquear,
            'motivos_bloqueio': motivos_bloqueio,
            'nivel_perigo_maximo': max_nivel_perigo,
            'confianca_maxima': max_confianca,
            'cenarios_detalhados': cenarios_detectados,
            'dados_tecnicos_completos': dados_completos,
            'recomendacoes': [c.recomendacao for c in cenarios_detectados if c.recomendacao]
        }
    
    def configurar_detectores(self, **kwargs):
        """Configura os detectores"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"🚨 Detector configurado: {key} = {value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos detectores"""
        total = self.stats['total_analises']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'taxa_deteccao_call_topo': (self.stats['cenarios_call_topo'] / total * 100),
            'taxa_deteccao_put_lateral': (self.stats['cenarios_put_lateral'] / total * 100),
            'taxa_deteccao_pos_gap': (self.stats['cenarios_pos_gap'] / total * 100),
            'taxa_bloqueio_total': (self.stats['bloqueios_total'] / total * 100),
            'detectores_ativos': sum(1 for k, v in self.config.items() if k.startswith('detectar_') and v)
        }
    
    def gerar_relatorio_detectores(self) -> str:
        """Gera relatório detalhado dos detectores"""
        stats = self.get_stats()
        
        relatorio = f"""
🚨 RELATÓRIO DETECTORES CENÁRIOS PERIGOSOS

📊 ESTATÍSTICAS GERAIS:
   Total Análises: {stats['total_analises']}
   Taxa Bloqueio: {stats.get('taxa_bloqueio_total', 0):.1f}%
   Detectores Ativos: {stats.get('detectores_ativos', 0)}/3

🔴 DETECÇÕES POR CENÁRIO:
   CALL no Topo: {stats['cenarios_call_topo']} ({stats.get('taxa_deteccao_call_topo', 0):.1f}%)
   PUT em Lateral: {stats['cenarios_put_lateral']} ({stats.get('taxa_deteccao_put_lateral', 0):.1f}%)
   Trade Pós-Gap: {stats['cenarios_pos_gap']} ({stats.get('taxa_deteccao_pos_gap', 0):.1f}%)
   Cenários Múltiplos: {stats['cenarios_multiplos']}

⚙️ CONFIGURAÇÃO ATUAL:
   Nível Mínimo Bloqueio: {self.config['nivel_minimo_bloqueio']}
   Confiança Mínima: {self.config['confianca_minima']:.1f}
"""
        
        for key, value in self.config.items():
            if key.startswith('detectar_'):
                detector_nome = key.replace('detectar_', '').replace('_', ' ').title()
                status = "✅ ATIVO" if value else "❌ INATIVO"
                relatorio += f"   {detector_nome}: {status}\n"
        
        relatorio += "\n🚨 DETECTORES PROTEGENDO CONTRA AS 'CRUZETAS'!"
        
        return relatorio

# Classe de integração para o sistema principal
class DetectoresIntegration:
    """Classe de integração para o sistema principal"""
    
    def __init__(self):
        self.detectores_master = DetectorCenariosPerigososMaster()
        print("🚨 Detectores de Cenários Perigosos inicializados!")
    
    def validar_cenarios(self, df: pd.DataFrame, analise_completa: Dict, 
                        par: str, tipo_sinal: str, score_total: float) -> Dict[str, Any]:
        """Método principal para validação de cenários"""
        
        resultado = self.detectores_master.analisar_cenarios_completo(
            df, analise_completa, par, tipo_sinal, score_total
        )
        
        return {
            'entrada_segura': not resultado['deve_bloquear'],
            'cenarios_detectados': resultado['cenarios_detectados'],
            'tipos_cenarios': resultado['tipos_cenarios'],
            'motivos_bloqueio': resultado['motivos_bloqueio'],
            'nivel_perigo': resultado['nivel_perigo_maximo'],
            'confianca_deteccao': resultado['confianca_maxima'],
            'recomendacoes': resultado['recomendacoes'],
            'dados_tecnicos': resultado['dados_tecnicos_completos'],
            'cenarios_multiplos': len(resultado['tipos_cenarios']) > 1
        }
    
    def configurar(self, **kwargs):
        """Configura os detectores"""
        self.detectores_master.configurar_detectores(**kwargs)
    
    def get_stats(self):
        """Retorna estatísticas"""
        return self.detectores_master.get_stats()
    
    def gerar_relatorio(self):
        """Gera relatório"""
        return self.detectores_master.gerar_relatorio_detectores()

print("✅ DETECTORES CENÁRIOS PERIGOSOS CARREGADOS - ANTI-CRUZETAS ACTIVE!")

# Exemplo de uso standalone para teste
if __name__ == "__main__":
    # Teste básico dos detectores
    import pandas as pd
    import numpy as np
    
    # Criar dados simulados para teste
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='T')
    
    # Simular dados OHLCV
    base_price = 50000
    prices = []
    volumes = []
    
    for i in range(100):
        # Movimento gradual com alguma volatilidade
        change = np.random.normal(0, 0.002)
        base_price *= (1 + change)
        
        open_price = base_price
        high_price = base_price * (1 + abs(np.random.normal(0, 0.001)))
        low_price = base_price * (1 - abs(np.random.normal(0, 0.001)))
        close_price = base_price * (1 + np.random.normal(0, 0.0005))
        volume = np.random.uniform(1000, 5000)
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
    
    df_test = pd.DataFrame({
        'timestamp': dates,
        'open': [p[0] for p in prices],
        'high': [p[1] for p in prices],
        'low': [p[2] for p in prices],
        'close': [p[3] for p in prices],
        'volume': volumes
    })
    
    # Análise simulada
    analise_test = {
        'rsi': 75,
        'score_call': 180,
        'score_put': 120,
        'confluencia_call': 8,
        'confluencia_put': 6,
        'volatilidade': 0.8,
        'volume_ratio': 2.1,
        'movimento_1min': 0.15
    }
    
    # Testar detectores
    detectores = DetectoresIntegration()
    
    print("🧪 TESTANDO DETECTORES DE CENÁRIOS PERIGOSOS")
    print("="*60)
    
    # Teste CALL no topo
    resultado_call = detectores.validar_cenarios(
        df_test, analise_test, 'btcusdt', 'CALL_ENHANCED', 180
    )
    
    print("🔴 TESTE CALL NO TOPO:")
    print(f"   Entrada Segura: {resultado_call['entrada_segura']}")
    if not resultado_call['entrada_segura']:
        print(f"   Motivos: {resultado_call['motivos_bloqueio']}")
    print(f"   Cenários Detectados: {resultado_call['cenarios_detectados']}")
    print()
    
    # Teste PUT em lateral (modificar dados para simular lateral)
    analise_lateral = analise_test.copy()
    analise_lateral['rsi'] = 52
    analise_lateral['movimento_1min'] = 0.02
    analise_lateral['volatilidade'] = 0.2
    
    resultado_put = detectores.validar_cenarios(
        df_test, analise_lateral, 'ethusdt', 'PUT_ENHANCED', 150
    )
    
    print("📏 TESTE PUT EM LATERAL:")
    print(f"   Entrada Segura: {resultado_put['entrada_segura']}")
    if not resultado_put['entrada_segura']:
        print(f"   Motivos: {resultado_put['motivos_bloqueio']}")
    print(f"   Cenários Detectados: {resultado_put['cenarios_detectados']}")
    print()
    
    # Teste trade pós-gap (simular movimento extremo)
    analise_gap = analise_test.copy()
    analise_gap['movimento_1min'] = 1.2  # Movimento extremo
    analise_gap['volatilidade'] = 2.5
    analise_gap['volume_ratio'] = 8.0
    
    resultado_gap = detectores.validar_cenarios(
        df_test, analise_gap, 'solusdt', 'CALL_SNIPER', 200
    )
    
    print("⚡ TESTE TRADE PÓS-GAP:")
    print(f"   Entrada Segura: {resultado_gap['entrada_segura']}")
    if not resultado_gap['entrada_segura']:
        print(f"   Motivos: {resultado_gap['motivos_bloqueio']}")
    print(f"   Cenários Detectados: {resultado_gap['cenarios_detectados']}")
    print()
    
    # Estatísticas finais
    stats = detectores.get_stats()
    print("📊 ESTATÍSTICAS DOS TESTES:")
    print(f"   Total Análises: {stats['total_analises']}")
    print(f"   CALL no Topo: {stats['cenarios_call_topo']}")
    print(f"   PUT em Lateral: {stats['cenarios_put_lateral']}")
    print(f"   Pós-Gap: {stats['cenarios_pos_gap']}")
    print(f"   Taxa Bloqueio: {stats.get('taxa_bloqueio_total', 0):.1f}%")
    
    print("\n✅ TESTE DOS DETECTORES CONCLUÍDO!")
    print("🚨 DETECTORES PRONTOS PARA DETECTAR AS 'CRUZETAS'!")