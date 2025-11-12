#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤖 IA PREDICTOR - CIPHER ROYAL SUPREME ENHANCED + AI ANTI-LOSS 🤖
💎 SISTEMA DE PREDIÇÃO EM TEMPO REAL PARA VALIDAÇÃO DE SINAIS
🔥 CARREGA MODELOS TREINADOS E VALIDA SINAIS ANTES DA EMISSÃO
🎯 EVITA LOSSES NAS "CRUZETAS" IDENTIFICADAS PELOS MODELOS
"""

import numpy as np
import pandas as pd
import pickle
import time
import datetime
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PredictionResult:
    """Resultado da predição da IA"""
    should_emit: bool
    confidence: float
    model_predictions: Dict[str, float]
    risk_level: str  # LOW, MEDIUM, HIGH
    block_reason: Optional[str]
    ai_score_adjustment: float

class AIPredictorAntiLoss:
    
    def __init__(self, models_path: str = 'ai_models/'):
        self.models_path = models_path
        self.models = {}
        self.scaler = None
        self.metadata = {}
        self.is_loaded = False
        
        # 🔧 CORREÇÃO CRÍTICA: Thresholds mais realistas (funcionava antes do ML)
        self.confidence_thresholds = {
            'conservative': 0.45,  # Era 0.75 - MUITO restritivo 
            'moderate': 0.35,      # Era 0.65 - MUITO restritivo
            'aggressive': 0.25     # Era 0.55 - MUITO restritivo
        }
        
        # Pesos dos modelos
        self.model_weights = {
            'xgboost': 0.6,
            'random_forest': 0.4
        }
        
        # 🔧 CORREÇÃO: Modo fallback mais permissivo
        self.fallback_mode = True
        
        self.load_models()
        print("🤖 IA Predictor Anti-Loss inicializado!")
    
    def load_models(self) -> bool:
        """Carrega os modelos treinados"""
        try:
            if not os.path.exists(self.models_path):
                print(f"⚠️ Pasta de modelos não encontrada: {self.models_path}")
                print("💡 Sistema funcionará em modo fallback (como antes do ML)")
                self.fallback_mode = True
                return False
            
            # Carregar XGBoost
            xgb_path = os.path.join(self.models_path, 'xgboost_model.pkl')
            if os.path.exists(xgb_path):
                try:
                    with open(xgb_path, 'rb') as f:
                        self.models['xgboost'] = pickle.load(f)
                    print("✅ Modelo XGBoost carregado")
                except Exception as e:
                    print(f"⚠️ Erro carregando XGBoost: {e}")
            
            # Carregar Random Forest
            rf_path = os.path.join(self.models_path, 'random_forest_model.pkl')
            if os.path.exists(rf_path):
                try:
                    with open(rf_path, 'rb') as f:
                        self.models['random_forest'] = pickle.load(f)
                    print("✅ Modelo Random Forest carregado")
                except Exception as e:
                    print(f"⚠️ Erro carregando Random Forest: {e}")
            
            # Carregar Scaler
            scaler_path = os.path.join(self.models_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print("✅ Scaler carregado")
                except Exception as e:
                    print(f"⚠️ Erro carregando Scaler: {e}")
            
            # Carregar Metadata
            metadata_path = os.path.join(self.models_path, 'metadata.pkl')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)
                    print("✅ Metadata carregada")
                except Exception as e:
                    print(f"⚠️ Erro carregando Metadata: {e}")
            
            # 🔧 CORREÇÃO: Só considera carregado se TEM modelos E scaler
            if self.models and self.scaler:
                self.is_loaded = True
                self.fallback_mode = False
                print("🚀 IA Anti-Loss carregada e pronta para uso!")
                return True
            else:
                print("⚠️ Modelos incompletos - Sistema funcionará em modo fallback")
                self.fallback_mode = True
                return False
                
        except Exception as e:
            print(f"❌ Erro carregando modelos: {e}")
            print("⚠️ Sistema funcionará em modo fallback (como antes do ML)")
            self.fallback_mode = True
            return False
    
    def validar_sinal(self, df: pd.DataFrame, analise_completa: Dict, 
                     par: str, tipo_sinal: str, score_total: float, 
                     confluencia_count: int, modo: str = 'moderate') -> PredictionResult:
        """Valida um sinal usando os modelos IA"""
        
        # 🔧 CORREÇÃO CRÍTICA: Se não carregou ou erro, PERMITE o sinal (como funcionava antes)
        if not self.is_loaded or self.fallback_mode:
            return PredictionResult(
                should_emit=True,  # PERMITE sinal
                confidence=0.6,    # Confiança média
                model_predictions={},
                risk_level='MEDIUM',
                block_reason=None,
                ai_score_adjustment=0  # Sem ajuste
            )
        
        try:
            # Extrair features
            features = self.extrair_features_tempo_real(df, analise_completa, par, tipo_sinal)
            
            if features is None:
                # Sem features válidas - PERMITE sinal
                return PredictionResult(
                    should_emit=True,
                    confidence=0.5,
                    model_predictions={},
                    risk_level='MEDIUM',
                    block_reason=None,
                    ai_score_adjustment=0
                )
            
            # Normalizar features
            features_scaled = self.scaler.transform(features)
            
            # Predições dos modelos
            predictions = {}
            confidence_scores = {}
            
            for model_name, model in self.models.items():
                if model is not None:
                    try:
                        pred_proba = model.predict_proba(features_scaled)[0]
                        win_probability = pred_proba[1] if len(pred_proba) > 1 else 0.6  # Default 60%
                        
                        predictions[model_name] = win_probability
                        confidence_scores[model_name] = max(pred_proba) if len(pred_proba) > 1 else 0.6
                    except Exception as e:
                        print(f"⚠️ Erro modelo {model_name}: {e}")
                        # Continua sem este modelo
                        continue
            
            # Ensemble prediction (média ponderada)
            if predictions:
                ensemble_prediction = sum(
                    predictions[model] * self.model_weights.get(model, 1.0)
                    for model in predictions
                ) / sum(self.model_weights.get(model, 1.0) for model in predictions)
                
                ensemble_confidence = sum(
                    confidence_scores[model] * self.model_weights.get(model, 1.0)
                    for model in confidence_scores
                ) / sum(self.model_weights.get(model, 1.0) for model in confidence_scores)
            else:
                # 🔧 CORREÇÃO: Se nenhum modelo funcionou, PERMITE sinal
                ensemble_prediction = 0.6  # 60% - acima do threshold
                ensemble_confidence = 0.6
            
            # Determinar se deve emitir o sinal
            threshold = self.confidence_thresholds[modo]
            should_emit = ensemble_prediction >= threshold
            
            # 🔧 CORREÇÃO ADICIONAL: Se score original é muito alto, força permissão
            if score_total >= 75 and confluencia_count >= 8:
                should_emit = True  # Score muito alto sempre passa
            
            # Determinar nível de risco
            if ensemble_prediction >= 0.7:
                risk_level = 'LOW'
            elif ensemble_prediction >= 0.5:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            # Motivo do bloqueio se aplicável
            block_reason = None
            if not should_emit:
                block_reason = f"IA Prediction: {ensemble_prediction:.3f} < threshold {threshold:.3f}"
            
            # 🔧 CORREÇÃO: Ajuste de score mais conservador
            if ensemble_prediction > 0.65:
                ai_score_adjustment = min(15, (ensemble_prediction - 0.65) * 50)  # Max +15
            elif ensemble_prediction < 0.35:
                ai_score_adjustment = -min(20, (0.35 - ensemble_prediction) * 100)  # Max -20
            else:
                ai_score_adjustment = 0
            
            return PredictionResult(
                should_emit=should_emit,
                confidence=ensemble_confidence,
                model_predictions=predictions,
                risk_level=risk_level,
                block_reason=block_reason,
                ai_score_adjustment=ai_score_adjustment
            )
            
        except Exception as e:
            print(f"⚠️ Erro na validação IA: {e}")
            # 🔧 CORREÇÃO: Em caso de erro, PERMITE sinal (como funcionava antes)
            return PredictionResult(
                should_emit=True,
                confidence=0.5,
                model_predictions={},
                risk_level='MEDIUM',
                block_reason=None,
                ai_score_adjustment=0
            )
    
    def extrair_features_tempo_real(self, df: pd.DataFrame, analise_completa: Dict, 
                                   par: str, tipo_sinal: str) -> Optional[np.ndarray]:
        """Extrai features em tempo real do sinal atual"""
        try:
            if len(df) < 20:
                return None
            
            # Features básicas mais robustas
            closes = df['close'].values
            volumes = df['volume'].values
            current_price = closes[-1]
            
            # RSI
            rsi = self.calcular_rsi_simples(closes)
            
            # EMAs
            ema_9 = self.calcular_ema_simples(closes, 9)
            ema_21 = self.calcular_ema_simples(closes, 21)
            
            # Dados da análise
            volume_ratio = analise_completa.get('volume_ratio', 1.0)
            volatilidade = analise_completa.get('volatilidade', 0.3)
            
            # Score e confluência
            score_call = analise_completa.get('score_call', 0)
            score_put = analise_completa.get('score_put', 0)
            confluencia_call = analise_completa.get('confluencia_call', 0)
            confluencia_put = analise_completa.get('confluencia_put', 0)
            
            # Usar score da direção correta
            if 'CALL' in tipo_sinal:
                score_final = score_call
                confluencia_final = confluencia_call
            else:
                score_final = score_put
                confluencia_final = confluencia_put
            
            # Hora atual
            hora_atual = datetime.datetime.now().hour
            
            # Codificar par
            par_encoding = {
                'btcusdt': 0, 'ethusdt': 1, 'solusdt': 2, 'xrpusdt': 3, 'adausdt': 4
            }
            par_encoded = par_encoding.get(par.lower(), 0)
            
            # 🔧 CORREÇÃO: Features completas para 26 features (igual ao treino)
            # Análise adicional para completar features
            macd_info = analise_completa.get('macd_info', {})
            bb_info = analise_completa.get('bb_info', {})
            
            # Features de suporte/resistência
            sr_info = self.analisar_suporte_resistencia_simples(df, current_price)
            
            # Features de tendência
            tendencia_info = self.analisar_tendencias_simples(closes)
            
            # Features de momentum
            momentum_info = self.analisar_momentum_simples(closes, volumes)
            
            features = [
                # Features básicas (1-10)
                min(max(rsi, 0), 100) / 100,  # 1. RSI normalizado 0-1
                min(max(ema_9 / current_price, 0.8), 1.2),  # 2. EMA9 ratio limitado
                min(max(ema_21 / current_price, 0.8), 1.2),  # 3. EMA21 ratio limitado
                min(max(volume_ratio, 0.1), 10.0),  # 4. Volume ratio limitado
                min(max(volatilidade, 0.01), 1.0),  # 5. Volatilidade limitada
                min(max(score_final, 0), 100) / 100,  # 6. Score normalizado 0-1
                min(max(confluencia_final, 0), 20) / 20,  # 7. Confluência normalizada 0-1
                (hora_atual % 24) / 24,  # 8. Hora normalizada 0-1
                par_encoded / 4,  # 9. Par normalizado 0-1
                1 if 'CALL' in tipo_sinal else 0,  # 10. Direção binária
                
                # Features técnicas (11-20)
                min(max(macd_info.get('linha', 0), -1), 1),  # 11. MACD linha
                min(max(macd_info.get('sinal', 0), -1), 1),  # 12. MACD sinal
                min(max(bb_info.get('posicao', 0.5), 0), 1),  # 13. BB posição
                min(max(sr_info.get('distancia_sr', 0.1), 0), 1),  # 14. Distância S/R
                min(max(sr_info.get('forca_sr', 1), 0), 10) / 10,  # 15. Força S/R
                min(max(tendencia_info.get('tendencia_curta', 0), -1), 1),  # 16. Tendência curta
                min(max(tendencia_info.get('tendencia_media', 0), -1), 1),  # 17. Tendência média
                min(max(momentum_info.get('momentum', 0), -1), 1),  # 18. Momentum
                min(max(momentum_info.get('aceleracao', 0), -1), 1),  # 19. Aceleração
                min(max(analise_completa.get('movimento_1min', 0), -5), 5) / 5,  # 20. Movimento 1min
                
                # Features extras (21-26) - Valores calculados ou padrão
                min(max(len(closes) / 200, 0), 1),  # 21. Dados disponíveis
                min(max(current_price / 100000, 0), 1),  # 22. Preço normalizado
                min(max(np.std(closes[-5:]) / current_price, 0), 0.1) * 10,  # 23. Volatilidade 5v
                min(max(analise_completa.get('enhanced_weight_aplicado', 1), 0.5), 1.5) - 0.5,  # 24. Enhanced weight
                1 if analise_completa.get('auto_calibrador_usado', False) else 0,  # 25. Auto calibrador
                min(max(analise_completa.get('price_action_score_call' if 'CALL' in tipo_sinal else 'price_action_score_put', 0), 0), 100) / 100  # 26. Price action
            ]
            
            # Verificar se features são válidas
            features_array = np.array(features)
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                return None
            
            return features_array.reshape(1, -1)
            
        except Exception as e:
            print(f"⚠️ Erro extraindo features: {e}")
            return None
    
    def analisar_suporte_resistencia_simples(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Análise simples de suporte/resistência"""
        try:
            if len(df) < 20:
                return {'distancia_sr': 0.1, 'forca_sr': 1.0}
            
            highs = df['high'].values[-20:]
            lows = df['low'].values[-20:]
            
            # Encontrar níveis próximos
            niveis_proximos = []
            for h in highs:
                if abs(h - current_price) / current_price < 0.02:
                    niveis_proximos.append(h)
            for l in lows:
                if abs(l - current_price) / current_price < 0.02:
                    niveis_proximos.append(l)
            
            if niveis_proximos:
                distancia_media = np.mean([abs(n - current_price) / current_price for n in niveis_proximos])
                forca = len(niveis_proximos)
            else:
                distancia_media = 0.1
                forca = 1.0
            
            return {
                'distancia_sr': min(distancia_media, 1.0),
                'forca_sr': min(forca, 10.0)
            }
        except:
            return {'distancia_sr': 0.1, 'forca_sr': 1.0}
    
    def analisar_tendencias_simples(self, closes: np.ndarray) -> Dict:
        """Análise simples de tendências"""
        try:
            if len(closes) < 10:
                return {'tendencia_curta': 0, 'tendencia_media': 0}
            
            # Tendência curta (5 velas)
            if len(closes) >= 5:
                tend_curta = (closes[-1] - closes[-5]) / closes[-5]
            else:
                tend_curta = 0
            
            # Tendência média (10 velas)
            if len(closes) >= 10:
                tend_media = (closes[-1] - closes[-10]) / closes[-10]
            else:
                tend_media = 0
            
            return {
                'tendencia_curta': max(-1, min(1, tend_curta * 50)),  # Normalizar
                'tendencia_media': max(-1, min(1, tend_media * 25))   # Normalizar
            }
        except:
            return {'tendencia_curta': 0, 'tendencia_media': 0}
    
    def analisar_momentum_simples(self, closes: np.ndarray, volumes: np.ndarray) -> Dict:
        """Análise simples de momentum"""
        try:
            if len(closes) < 5:
                return {'momentum': 0, 'aceleracao': 0}
            
            # Momentum (velocidade do movimento)
            movimento_recente = np.mean(np.diff(closes[-3:]))
            movimento_anterior = np.mean(np.diff(closes[-6:-3]))
            
            if abs(movimento_anterior) > 0:
                momentum = movimento_recente / abs(movimento_anterior)
            else:
                momentum = 0
            
            # Aceleração (mudança do momentum)
            if len(closes) >= 6:
                aceleracao = movimento_recente - movimento_anterior
                aceleracao = aceleracao / closes[-1] * 1000  # Normalizar
            else:
                aceleracao = 0
            
            return {
                'momentum': max(-1, min(1, momentum)),
                'aceleracao': max(-1, min(1, aceleracao))
            }
        except:
            return {'momentum': 0, 'aceleracao': 0}
    
    def calcular_rsi_simples(self, prices: np.ndarray, periodo: int = 14) -> float:
        """Calcula RSI simples"""
        if len(prices) < periodo + 1:
            return 50.0
        
        try:
            deltas = np.diff(prices)
            ganhos = np.where(deltas > 0, deltas, 0)
            perdas = np.where(deltas < 0, -deltas, 0)
            
            ganho_medio = np.mean(ganhos[-periodo:])
            perda_media = np.mean(perdas[-periodo:])
            
            if perda_media == 0:
                return 100.0
            
            rs = ganho_medio / perda_media
            rsi = 100 - (100 / (1 + rs))
            
            return max(0, min(100, rsi))  # Garantir 0-100
        except:
            return 50.0
    
    def calcular_ema_simples(self, prices: np.ndarray, periodo: int) -> float:
        """Calcula EMA simples"""
        if len(prices) < periodo:
            return np.mean(prices) if len(prices) > 0 else 0
        
        try:
            multiplicador = 2 / (periodo + 1)
            ema = prices[0]
            for preco in prices[1:]:
                ema = (preco * multiplicador) + (ema * (1 - multiplicador))
            return ema
        except:
            return prices[-1] if len(prices) > 0 else 0
    
    def get_model_status(self) -> Dict[str, Any]:
        """Retorna status dos modelos carregados"""
        return {
            'is_loaded': self.is_loaded,
            'models_available': list(self.models.keys()),
            'scaler_loaded': self.scaler is not None,
            'metadata_loaded': bool(self.metadata),
            'models_path': self.models_path,
            'fallback_mode': self.fallback_mode
        }
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, List]:
        """Retorna as features mais importantes dos modelos"""
        if not self.metadata or 'feature_importance' not in self.metadata:
            return {}
        
        result = {}
        for model_name, features in self.metadata['feature_importance'].items():
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
            result[model_name] = sorted_features[:top_n]
        
        return result
    
    def update_prediction_feedback(self, prediction_id: str, actual_result: str):
        """Atualiza feedback de predições para retreino futuro"""
        # Esta função seria implementada para coletar feedback
        # e melhorar os modelos ao longo do tempo
        pass

# Classe de conveniência para integração
class AIValidatorIntegration:
    """Classe de integração para o sistema principal"""
    
    def __init__(self):
        self.predictor = AIPredictorAntiLoss()
        self.modo_validacao = 'moderate'  # conservative, moderate, aggressive
        self.stats = {
            'sinais_validados': 0,
            'sinais_bloqueados': 0,
            'bloqueios_por_ia': 0,
            'ajustes_score_aplicados': 0
        }
    
    def validar_entrada(self, df: pd.DataFrame, analise_completa: Dict, 
                       par: str, tipo_sinal: str, score_total: float, 
                       confluencia_count: int) -> Dict[str, Any]:
        """Método principal para validação de entrada"""
        
        # Validar com IA
        prediction = self.predictor.validar_sinal(
            df, analise_completa, par, tipo_sinal, 
            score_total, confluencia_count, self.modo_validacao
        )
        
        # Atualizar estatísticas
        self.stats['sinais_validados'] += 1
        
        if not prediction.should_emit:
            self.stats['sinais_bloqueados'] += 1
            if 'IA Prediction' in (prediction.block_reason or ''):
                self.stats['bloqueios_por_ia'] += 1
        
        if abs(prediction.ai_score_adjustment) > 0:
            self.stats['ajustes_score_aplicados'] += 1
        
        return {
            'entrada_segura': prediction.should_emit,
            'ia_confidence': prediction.confidence,
            'ia_predictions': prediction.model_predictions,
            'risk_level': prediction.risk_level,
            'motivo_bloqueio': prediction.block_reason,
            'score_adjustment': prediction.ai_score_adjustment,
            'ia_ativa': self.predictor.is_loaded
        }
    
    def configurar_modo(self, modo: str):
        """Configura modo de validação: conservative, moderate, aggressive"""
        if modo in ['conservative', 'moderate', 'aggressive']:
            self.modo_validacao = modo
            print(f"🤖 Modo IA configurado: {modo.upper()}")
        else:
            print(f"⚠️ Modo inválido: {modo}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da IA"""
        total = self.stats['sinais_validados']
        return {
            **self.stats,
            'taxa_bloqueio': (self.stats['sinais_bloqueados'] / total * 100) if total > 0 else 0,
            'taxa_bloqueio_ia': (self.stats['bloqueios_por_ia'] / total * 100) if total > 0 else 0,
            'model_status': self.predictor.get_model_status()
        }

print("✅ IA PREDICTOR ANTI-LOSS CARREGADO - THRESHOLDS CORRIGIDOS!")
print("🔧 MODO FALLBACK ATIVO - SISTEMA FUNCIONARÁ COMO ANTES SE ML FALHAR!")