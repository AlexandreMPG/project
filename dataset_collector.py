#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 DATASET COLLECTOR - CIPHER ROYAL SUPREME ENHANCED + AI 📊
💎 COLETA AUTOMÁTICA DE DADOS PARA TREINO DA IA ANTI-LOSS
🔥 FEATURES EXTRAÇÃO + LABELING + PREPARAÇÃO DATASET
🎯 COLETA CONTÍNUA DE DADOS DE MERCADO E RESULTADOS
"""

import numpy as np
import pandas as pd
import requests
import sqlite3
import time
import datetime
import threading
import schedule
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import json

@dataclass
class MarketSnapshot:
    """Snapshot do mercado no momento do sinal"""
    timestamp: int
    par: str
    # OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Indicadores técnicos
    rsi_14: float
    ema_9: float
    ema_21: float
    macd_line: float
    macd_signal: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float
    
    # Volume e volatilidade
    volume_sma_20: float
    volume_ratio: float
    volatility_20: float
    atr_14: float
    
    # Suporte/Resistência
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    support_strength: int
    resistance_strength: int
    distance_to_sr: float
    
    # Padrões de velas
    candle_pattern: str
    upper_shadow_ratio: float
    lower_shadow_ratio: float
    body_ratio: float
    
    # Contexto de tendência
    trend_5m: int  # -1, 0, 1
    trend_15m: int
    trend_1h: int
    price_position_in_range: float  # 0-1
    
    # Momentum
    momentum_10: float
    roc_10: float
    
    # Horário e contexto
    hour_of_day: int
    day_of_week: int
    
    # Score técnico original
    technical_score: float
    confluence_count: int

@dataclass
class FeatureSet:
    """Conjunto completo de features para ML"""
    # Features normalizadas para ML
    rsi_normalized: float
    ema_ratio_9_21: float
    macd_divergence: float
    bb_position: float
    volume_ratio_log: float
    volatility_percentile: float
    
    # Features categóricas
    candle_pattern_encoded: int
    trend_alignment: int  # -3 a 3
    hour_category: int  # 0-5 (categorias de horário)
    
    # Features de contexto
    sr_proximity_score: float
    momentum_strength: float
    market_phase: int  # 0-4 (acumulação, alta, distribuição, baixa, lateral)
    
    # Target
    signal_type: str
    actual_result: Optional[str]  # WIN_M1, WIN_GALE, LOSS

class TechnicalAnalyzer:
    """Analisador técnico para extração de features"""
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calcula RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> float:
        """Calcula EMA"""
        if len(prices) < period:
            return np.mean(prices) if len(prices) > 0 else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calcula MACD"""
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = TechnicalAnalyzer.calculate_ema(prices, fast)
        ema_slow = TechnicalAnalyzer.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Simplificado: usar a própria linha MACD como sinal
        macd_signal = macd_line * 0.9  # Aproximação
        
        return macd_line, macd_signal
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float, float]:
        """Calcula Bollinger Bands"""
        if len(prices) < period:
            sma = np.mean(prices) if len(prices) > 0 else 0
            return sma, sma, sma, 0.5
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        current_price = prices[-1]
        position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        
        return upper, sma, lower, position
    
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Tuple[Optional[float], Optional[float], int, int]:
        """Detecta suporte e resistência"""
        if len(df) < lookback:
            return None, None, 0, 0
        
        highs = df['high'].values[-lookback:]
        lows = df['low'].values[-lookback:]
        current_price = df['close'].iloc[-1]
        
        # Encontrar pivôs de suporte
        supports = []
        for i in range(2, len(lows) - 2):
            if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                supports.append(lows[i])
        
        # Encontrar pivôs de resistência
        resistances = []
        for i in range(2, len(highs) - 2):
            if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                resistances.append(highs[i])
        
        # Encontrar mais próximos
        nearest_support = None
        nearest_resistance = None
        support_strength = 0
        resistance_strength = 0
        
        if supports:
            supports_below = [s for s in supports if s <= current_price]
            if supports_below:
                nearest_support = max(supports_below)
                support_strength = sum(1 for s in supports if abs(s - nearest_support) / nearest_support < 0.002)
        
        if resistances:
            resistances_above = [r for r in resistances if r >= current_price]
            if resistances_above:
                nearest_resistance = min(resistances_above)
                resistance_strength = sum(1 for r in resistances if abs(r - nearest_resistance) / nearest_resistance < 0.002)
        
        return nearest_support, nearest_resistance, support_strength, resistance_strength
    
    @staticmethod
    def analyze_candle_pattern(df: pd.DataFrame) -> Tuple[str, float, float, float]:
        """Analisa padrão da vela atual"""
        if len(df) < 1:
            return "unknown", 0.0, 0.0, 0.0
        
        current = df.iloc[-1]
        o, h, l, c = current['open'], current['high'], current['low'], current['close']
        
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range == 0:
            return "unknown", 0.0, 0.0, 0.0
        
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        body_ratio = body / total_range
        
        # Detectar padrões básicos
        if body_ratio < 0.1:
            pattern = "doji"
        elif lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.2:
            pattern = "hammer"
        elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.2:
            pattern = "shooting_star"
        elif body_ratio > 0.7:
            pattern = "marubozu"
        else:
            pattern = "normal"
        
        return pattern, upper_shadow_ratio, lower_shadow_ratio, body_ratio
    
    @staticmethod
    def calculate_trend(prices: np.ndarray, period: int) -> int:
        """Calcula tendência (-1, 0, 1)"""
        if len(prices) < period:
            return 0
        
        start_price = prices[-period]
        end_price = prices[-1]
        change_pct = (end_price - start_price) / start_price * 100
        
        if change_pct > 0.5:
            return 1  # Alta
        elif change_pct < -0.5:
            return -1  # Baixa
        else:
            return 0  # Lateral

class DatasetCollector:
    """Coletor principal de dados para IA"""
    
    def __init__(self, db_path: str = 'royal_supreme_enhanced.db'):
        self.db_path = db_path
        self.cache_dados = {}
        self.collecting = False
        self.collection_thread = None
        
        # Buffer para dados em tempo real
        self.market_buffer = deque(maxlen=1000)
        self.signals_buffer = deque(maxlen=100)
        
        self.analyzer = TechnicalAnalyzer()
        
        # Estatísticas
        self.stats = {
            'snapshots_coletados': 0,
            'features_extraidas': 0,
            'sinais_processados': 0,
            'datasets_gerados': 0
        }
        
        print("📊 Dataset Collector inicializado!")
    
    def buscar_dados_mercado(self, par: str, timeframe: str = '1m', limit: int = 200) -> pd.DataFrame:
        """Busca dados atuais do mercado"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={par.upper()}&interval={timeframe}&limit={limit}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                return df
            
        except Exception as e:
            print(f"⚠️ Erro buscando dados {par}: {e}")
        
        return pd.DataFrame()
    
    def extrair_snapshot_mercado(self, par: str, signal_timestamp: int) -> Optional[MarketSnapshot]:
        """Extrai snapshot completo do mercado"""
        try:
            df = self.buscar_dados_mercado(par, '1m', 100)
            
            if df.empty or len(df) < 50:
                return None
            
            # Dados OHLCV atuais
            current = df.iloc[-1]
            
            # Calcular indicadores
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            # RSI
            rsi = self.analyzer.calculate_rsi(closes)
            
            # EMAs
            ema_9 = self.analyzer.calculate_ema(closes, 9)
            ema_21 = self.analyzer.calculate_ema(closes, 21)
            
            # MACD
            macd_line, macd_signal = self.analyzer.calculate_macd(closes)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_position = self.analyzer.calculate_bollinger_bands(closes)
            
            # Volume
            volume_sma = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            volume_ratio = current['volume'] / volume_sma if volume_sma > 0 else 1.0
            
            # Volatilidade
            volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100 if len(closes) >= 20 else 0
            
            # ATR
            tr_values = []
            for i in range(1, min(15, len(df))):
                tr = max(
                    highs[-i] - lows[-i],
                    abs(highs[-i] - closes[-i-1]),
                    abs(lows[-i] - closes[-i-1])
                )
                tr_values.append(tr)
            atr = np.mean(tr_values) if tr_values else 0
            
            # Suporte/Resistência
            nearest_support, nearest_resistance, support_strength, resistance_strength = \
                self.analyzer.detect_support_resistance(df)
            
            current_price = current['close']
            distance_to_sr = 0.0
            if nearest_support and nearest_resistance:
                dist_support = abs(current_price - nearest_support) / current_price
                dist_resistance = abs(current_price - nearest_resistance) / current_price
                distance_to_sr = min(dist_support, dist_resistance)
            
            # Padrão de vela
            candle_pattern, upper_shadow_ratio, lower_shadow_ratio, body_ratio = \
                self.analyzer.analyze_candle_pattern(df)
            
            # Tendências
            trend_5m = self.analyzer.calculate_trend(closes, 5)
            trend_15m = self.analyzer.calculate_trend(closes, 15)
            trend_1h = self.analyzer.calculate_trend(closes, min(60, len(closes)))
            
            # Posição no range
            if len(closes) >= 20:
                range_high = np.max(closes[-20:])
                range_low = np.min(closes[-20:])
                price_position = (current_price - range_low) / (range_high - range_low) if range_high != range_low else 0.5
            else:
                price_position = 0.5
            
            # Momentum
            momentum = (current_price - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 else 0
            roc = (current_price - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 else 0
            
            # Contexto temporal
            dt = datetime.datetime.fromtimestamp(signal_timestamp)
            hour_of_day = dt.hour
            day_of_week = dt.weekday()
            
            return MarketSnapshot(
                timestamp=signal_timestamp,
                par=par,
                open=current['open'],
                high=current['high'],
                low=current['low'],
                close=current['close'],
                volume=current['volume'],
                rsi_14=rsi,
                ema_9=ema_9,
                ema_21=ema_21,
                macd_line=macd_line,
                macd_signal=macd_signal,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_position=bb_position,
                volume_sma_20=volume_sma,
                volume_ratio=volume_ratio,
                volatility_20=volatility,
                atr_14=atr,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                support_strength=support_strength,
                resistance_strength=resistance_strength,
                distance_to_sr=distance_to_sr,
                candle_pattern=candle_pattern,
                upper_shadow_ratio=upper_shadow_ratio,
                lower_shadow_ratio=lower_shadow_ratio,
                body_ratio=body_ratio,
                trend_5m=trend_5m,
                trend_15m=trend_15m,
                trend_1h=trend_1h,
                price_position_in_range=price_position,
                momentum_10=momentum,
                roc_10=roc,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                technical_score=0.0,  # Será preenchido externamente
                confluence_count=0    # Será preenchido externamente
            )
            
        except Exception as e:
            print(f"⚠️ Erro extraindo snapshot {par}: {e}")
            return None
    
    def converter_para_features(self, snapshot: MarketSnapshot) -> FeatureSet:
        """Converte snapshot em features normalizadas para ML"""
        try:
            # Normalizar RSI (0-100 -> 0-1)
            rsi_normalized = snapshot.rsi_14 / 100.0
            
            # Ratio EMAs
            ema_ratio = (snapshot.ema_9 / snapshot.ema_21) if snapshot.ema_21 > 0 else 1.0
            
            # MACD divergência
            macd_divergence = snapshot.macd_line - snapshot.macd_signal
            
            # Volume ratio log (para normalizar outliers)
            volume_ratio_log = np.log1p(snapshot.volume_ratio)
            
            # Volatilidade percentil (aproximação)
            volatility_percentile = min(snapshot.volatility_20 / 2.0, 1.0)  # Assumindo 2% como máximo normal
            
            # Encoding de padrão de vela
            pattern_encoding = {
                'doji': 0, 'hammer': 1, 'shooting_star': 2, 
                'marubozu': 3, 'normal': 4, 'unknown': 5
            }
            candle_pattern_encoded = pattern_encoding.get(snapshot.candle_pattern, 5)
            
            # Alinhamento de tendência (-3 a 3)
            trend_alignment = snapshot.trend_5m + snapshot.trend_15m + snapshot.trend_1h
            
            # Categoria de horário
            hour_categories = {
                (0, 5): 0,   # Madrugada
                (6, 11): 1,  # Manhã
                (12, 17): 2, # Tarde
                (18, 23): 3  # Noite
            }
            hour_category = 4  # Default
            for (start, end), category in hour_categories.items():
                if start <= snapshot.hour_of_day <= end:
                    hour_category = category
                    break
            
            # Score de proximidade S/R
            sr_proximity_score = 1.0 - min(snapshot.distance_to_sr * 100, 1.0)  # Quanto mais próximo, maior o score
            
            # Força do momentum
            momentum_strength = min(abs(snapshot.momentum_10) / 5.0, 1.0)  # Normalizar por 5%
            
            # Fase do mercado (simplificado)
            if snapshot.trend_5m == 1 and snapshot.trend_15m == 1:
                market_phase = 1  # Alta
            elif snapshot.trend_5m == -1 and snapshot.trend_15m == -1:
                market_phase = 3  # Baixa
            elif abs(snapshot.momentum_10) < 0.1:
                market_phase = 4  # Lateral
            else:
                market_phase = 0  # Indefinido
            
            return FeatureSet(
                rsi_normalized=rsi_normalized,
                ema_ratio_9_21=ema_ratio,
                macd_divergence=macd_divergence,
                bb_position=snapshot.bb_position,
                volume_ratio_log=volume_ratio_log,
                volatility_percentile=volatility_percentile,
                candle_pattern_encoded=candle_pattern_encoded,
                trend_alignment=trend_alignment,
                hour_category=hour_category,
                sr_proximity_score=sr_proximity_score,
                momentum_strength=momentum_strength,
                market_phase=market_phase,
                signal_type="",  # Será preenchido
                actual_result=None  # Será preenchido
            )
            
        except Exception as e:
            print(f"⚠️ Erro convertendo features: {e}")
            return None
    
    def coletar_sinal_emitido(self, sinal_data: Dict[str, Any]) -> bool:
        """Coleta dados quando um sinal é emitido"""
        try:
            # Extrair snapshot do mercado
            snapshot = self.extrair_snapshot_mercado(
                sinal_data['par'], 
                sinal_data['timestamp']
            )
            
            if snapshot is None:
                return False
            
            # Adicionar dados do sinal
            snapshot.technical_score = sinal_data.get('score', 0)
            snapshot.confluence_count = sinal_data.get('confluencia', 0)
            
            # Converter para features
            features = self.converter_para_features(snapshot)
            if features is None:
                return False
            
            features.signal_type = sinal_data['tipo']
            
            # Salvar no buffer
            self.market_buffer.append(snapshot)
            self.signals_buffer.append({
                'snapshot': snapshot,
                'features': features,
                'sinal_data': sinal_data,
                'resultado_pendente': True
            })
            
            self.stats['snapshots_coletados'] += 1
            self.stats['features_extraidas'] += 1
            
            print(f"📊 Snapshot coletado: {sinal_data['par']} {sinal_data['tipo']}")
            return True
            
        except Exception as e:
            print(f"⚠️ Erro coletando sinal: {e}")
            return False
    
    def atualizar_resultado_sinal(self, timestamp: int, par: str, resultado: str) -> bool:
        """Atualiza resultado de um sinal coletado"""
        try:
            # Buscar sinal no buffer
            for signal_data in self.signals_buffer:
                if (signal_data['snapshot'].timestamp == timestamp and 
                    signal_data['snapshot'].par == par and
                    signal_data['resultado_pendente']):
                    
                    # Atualizar resultado
                    signal_data['features'].actual_result = resultado
                    signal_data['resultado_pendente'] = False
                    
                    # Salvar no banco de dados
                    self.salvar_dataset_entry(signal_data)
                    
                    self.stats['sinais_processados'] += 1
                    print(f"📊 Resultado atualizado: {par} {resultado}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Erro atualizando resultado: {e}")
            return False
    
    def salvar_dataset_entry(self, signal_data: Dict) -> bool:
        """Salva entrada completa do dataset no banco"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Criar tabela se não existir
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_dataset (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    par TEXT,
                    signal_type TEXT,
                    result TEXT,
                    
                    -- Features ML
                    rsi_normalized REAL,
                    ema_ratio_9_21 REAL,
                    macd_divergence REAL,
                    bb_position REAL,
                    volume_ratio_log REAL,
                    volatility_percentile REAL,
                    candle_pattern_encoded INTEGER,
                    trend_alignment INTEGER,
                    hour_category INTEGER,
                    sr_proximity_score REAL,
                    momentum_strength REAL,
                    market_phase INTEGER,
                    
                    -- Raw market data (JSON)
                    market_snapshot TEXT,
                    
                    -- Metadata
                    technical_score REAL,
                    confluence_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            snapshot = signal_data['snapshot']
            features = signal_data['features']
            
            # Converter snapshot para JSON
            snapshot_json = json.dumps(asdict(snapshot))
            
            cursor.execute('''
                INSERT INTO ml_dataset (
                    timestamp, par, signal_type, result,
                    rsi_normalized, ema_ratio_9_21, macd_divergence, bb_position,
                    volume_ratio_log, volatility_percentile, candle_pattern_encoded,
                    trend_alignment, hour_category, sr_proximity_score,
                    momentum_strength, market_phase, market_snapshot,
                    technical_score, confluence_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp, snapshot.par, features.signal_type, features.actual_result,
                features.rsi_normalized, features.ema_ratio_9_21, features.macd_divergence,
                features.bb_position, features.volume_ratio_log, features.volatility_percentile,
                features.candle_pattern_encoded, features.trend_alignment, features.hour_category,
                features.sr_proximity_score, features.momentum_strength, features.market_phase,
                snapshot_json, snapshot.technical_score, snapshot.confluence_count
            ))
            
            conn.commit()
            conn.close()
            
            self.stats['datasets_gerados'] += 1
            return True
            
        except Exception as e:
            print(f"⚠️ Erro salvando dataset: {e}")
            return False
    
    def gerar_dataset_para_treino(self, min_samples: int = 100) -> pd.DataFrame:
        """Gera dataset preparado para treino"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM ml_dataset 
                WHERE result IS NOT NULL 
                AND result IN ('WIN_M1', 'WIN_GALE', 'LOSS')
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(min_samples * 2,))
            conn.close()
            
            if len(df) < min_samples:
                print(f"⚠️ Dados insuficientes: {len(df)} < {min_samples}")
                return pd.DataFrame()
            
            print(f"📊 Dataset gerado: {len(df)} amostras")
            return df
            
        except Exception as e:
            print(f"⚠️ Erro gerando dataset: {e}")
            return pd.DataFrame()
    
    def limpar_dados_antigos(self, dias: int = 7):
        """Remove dados antigos do dataset"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp_limite = int(time.time()) - (dias * 24 * 3600)
            
            cursor.execute('DELETE FROM ml_dataset WHERE timestamp < ?', (timestamp_limite,))
            
            linhas_removidas = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"🗑️ Dados antigos removidos: {linhas_removidas} entradas")
            
        except Exception as e:
            print(f"⚠️ Erro limpando dados: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do coletor"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total de entradas no dataset
            cursor.execute('SELECT COUNT(*) FROM ml_dataset')
            total_dataset = cursor.fetchone()[0]
            
            # Por resultado
            cursor.execute('''
                SELECT result, COUNT(*) 
                FROM ml_dataset 
                WHERE result IS NOT NULL 
                GROUP BY result
            ''')
            
            resultado_counts = dict(cursor.fetchall())
            conn.close()
            
            return {
                **self.stats,
                'total_dataset': total_dataset,
                'win_m1_count': resultado_counts.get('WIN_M1', 0),
                'win_gale_count': resultado_counts.get('WIN_GALE', 0),
                'loss_count': resultado_counts.get('LOSS', 0),
                'buffer_size': len(self.signals_buffer),
                'pendentes': sum(1 for s in self.signals_buffer if s.get('resultado_pendente', False))
            }
            
        except Exception as e:
            print(f"⚠️ Erro obtendo stats: {e}")
            return self.stats
    
    def iniciar_coleta_automatica(self, intervalo_minutos: int = 10):
        """Inicia coleta automática de dados de mercado"""
        def job():
            try:
                pares = ['btcusdt', 'ethusdt', 'solusdt', 'xrpusdt', 'adausdt']
                for par in pares:
                    # Coletar snapshot de mercado atual
                    snapshot = self.extrair_snapshot_mercado(par, int(time.time()))
                    if snapshot:
                        self.market_buffer.append(snapshot)
                
                print(f"📊 Coleta automática executada: {len(pares)} pares")
                
            except Exception as e:
                print(f"⚠️ Erro na coleta automática: {e}")
        
        # Agendar job
        schedule.every(intervalo_minutos).minutes.do(job)
        
        # Thread para executar schedule
        def run_schedule():
            while self.collecting:
                schedule.run_pending()
                time.sleep(1)
        
        self.collecting = True
        self.collection_thread = threading.Thread(target=run_schedule)
        self.collection_thread.start()
        
        print(f"📊 Coleta automática iniciada: intervalo {intervalo_minutos} min")
    
    def parar_coleta_automatica(self):
        """Para a coleta automática"""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        
        print("📊 Coleta automática parada")
    
    def gerar_relatorio_dataset(self) -> str:
        """Gera relatório do dataset coletado"""
        stats = self.get_stats()
        
        total_completos = stats['win_m1_count'] + stats['win_gale_count'] + stats['loss_count']
        win_rate = ((stats['win_m1_count'] + stats['win_gale_count']) / total_completos * 100) if total_completos > 0 else 0
        
        relatorio = f"""
📊 RELATÓRIO DATASET COLLECTOR

📈 ESTATÍSTICAS DE COLETA:
   Snapshots Coletados: {stats['snapshots_coletados']}
   Features Extraídas: {stats['features_extraidas']}
   Sinais Processados: {stats['sinais_processados']}
   Datasets Gerados: {stats['datasets_gerados']}

🗄️ DATASET ATUAL:
   Total Entradas: {stats['total_dataset']}
   Completas (com resultado): {total_completos}
   Win Rate Dataset: {win_rate:.1f}%

📊 DISTRIBUIÇÃO RESULTADOS:
   WIN M1: {stats['win_m1_count']}
   WIN GALE: {stats['win_gale_count']}
   LOSSES: {stats['loss_count']}

🔄 BUFFER STATUS:
   Buffer Size: {stats['buffer_size']}
   Pendentes: {stats['pendentes']}

📊 QUALIDADE DOS DADOS:
   {"✅ Pronto para treino" if total_completos >= 50 else "⚠️ Mais dados necessários"}
   {"✅ Distribuição balanceada" if stats['loss_count'] > 0 and (stats['win_m1_count'] + stats['win_gale_count']) > 0 else "⚠️ Distribuição desbalanceada"}

📊 DATASET COLLECTOR FUNCIONANDO!
"""
        
        return relatorio

# Classe de integração para o sistema principal
class DatasetCollectorIntegration:
    """Classe de integração para o sistema principal"""
    
    def __init__(self):
        self.collector = DatasetCollector()
        print("📊 Dataset Collector Integration inicializado!")
    
    def registrar_sinal_emitido(self, sinal_data: Dict[str, Any]) -> bool:
        """Registra quando um sinal é emitido"""
        return self.collector.coletar_sinal_emitido(sinal_data)
    
    def registrar_resultado_sinal(self, timestamp: int, par: str, resultado: str) -> bool:
        """Registra resultado de um sinal"""
        return self.collector.atualizar_resultado_sinal(timestamp, par, resultado)
    
    def get_dataset_para_treino(self, min_samples: int = 100) -> pd.DataFrame:
        """Obtém dataset para treino"""
        return self.collector.gerar_dataset_para_treino(min_samples)
    
    def iniciar_coleta_automatica(self, intervalo: int = 10):
        """Inicia coleta automática"""
        self.collector.iniciar_coleta_automatica(intervalo)
    
    def parar_coleta_automatica(self):
        """Para coleta automática"""
        self.collector.parar_coleta_automatica()
    
    def get_stats(self):
        """Retorna estatísticas"""
        return self.collector.get_stats()
    
    def gerar_relatorio(self):
        """Gera relatório"""
        return self.collector.gerar_relatorio_dataset()
    
    def limpar_dados_antigos(self, dias: int = 7):
        """Limpa dados antigos"""
        self.collector.limpar_dados_antigos(dias)

print("✅ DATASET COLLECTOR CARREGADO - AUTOMATIC DATA COLLECTION ACTIVE!")