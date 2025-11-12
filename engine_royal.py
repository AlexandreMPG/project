#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 ENGINE ROYAL SUPREME ENHANCED + DATABASE AI + IA ANTI-LOSS 👑
💎 ENGINE PRINCIPAL DO SISTEMA
🔥 GERAÇÃO DE SINAIS + VERIFICAÇÃO WIN/LOSS + COOLDOWNS + DATABASE + IA ANTI-LOSS
🤖 INTEGRAÇÃO COMPLETA COM SISTEMA IA PARA EVITAR LOSSES
"""

import requests
import pandas as pd
import numpy as np
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from config_royal import (ConfigRoyalSupremeEnhanced, TipoSinalRoyalSupremeEnhanced, 
                         StatusSinalRoyalSupremeEnhanced, MarketScenarioRoyalSupremeEnhanced, 
                         SurvivabilityMode, PARES_CRYPTO, TIMEZONE)
from database_manager import DatabaseManager
from analisador_completo import (AnalisadorCompletoV8RoyalSupremeEnhanced, SinalRoyalSupremeEnhanced, 
                                RelatoriosRoyalSupremeEnhanced)
from telegram_system import RoyalSessionManagerSupremeEnhanced, TelegramAdminCommandsEnhanced, RoyalTelegramSupremeEnhanced
from detectores_mercado import DetectorCenariosExtremosV8RoyalSupremeEnhanced

# 🤖 IA ANTI-LOSS INTEGRATION
try:
    from integration_hooks import hook_validar_sinal_antes_emissao, hook_registrar_resultado_operacao
    IA_ANTI_LOSS_AVAILABLE = True
    print("🤖 IA ANTI-LOSS CARREGADA E INTEGRADA COM SUCESSO!")
except ImportError:
    IA_ANTI_LOSS_AVAILABLE = False
    print("⚠️ IA Anti-Loss não disponível - sistema funcionará normalmente")

class EngineRoyalSupremeEnhanced:
    
    def __init__(self):
        # CORREÇÃO: Ordem correta de inicialização
        self.relatorios = RelatoriosRoyalSupremeEnhanced()
        self.db_manager = DatabaseManager()
        self.analisador = AnalisadorCompletoV8RoyalSupremeEnhanced()
        
        # CORREÇÃO CRÍTICA: Inicializar sistema de sobrevivência corretamente
        try:
            self.analisador.inicializar_sobrevivencia(self.relatorios, self.db_manager)
            self.sistema_sobrevivencia = self.analisador.sistema_sobrevivencia
        except Exception as e:
            print(f"⚠️ Erro inicializando sobrevivência: {e}")
            self.sistema_sobrevivencia = None
        self.detector_cenarios = DetectorCenariosExtremosV8RoyalSupremeEnhanced()
        
        self.session_manager = RoyalSessionManagerSupremeEnhanced()
        self.telegram = RoyalTelegramSupremeEnhanced()
        self.admin_commands = TelegramAdminCommandsEnhanced(self.relatorios)
        
        # Conectar database manager aos relatórios para comandos admin
        self.relatorios.db_manager = self.db_manager
        
        self.cache_dados = {}
        self.ultimo_sinal_par = {}
        self.sinais_por_hora = defaultdict(list)
        self.protecoes_ativas = {}
        self.cooldown_inteligente = {}
        self.cooldown_pos_win = {}
        self.cooldown_pos_loss = {}
        
        self.config_operacao = {
            'pares_selecionados': ['btcusdt', 'ethusdt', 'solusdt', 'xrpusdt', 'adausdt'],
            'modo_selecao': 'TODOS_PARES',
            'par_individual': None,
            'sniper_only': False
        }
        
        # 🤖 Estatísticas IA Anti-Loss
        self.ia_stats = {
            'sinais_analisados': 0,
            'sinais_bloqueados': 0,
            'ajustes_aplicados': 0,
            'calibrador_execucoes': 0  # 🔧 NOVO: Contador calibrador
        }
        
        sistema_ia_status = "✅ ATIVO" if IA_ANTI_LOSS_AVAILABLE else "❌ INDISPONÍVEL"
        print(f"👑 ENGINE ROYAL SUPREME ENHANCED + DATABASE AI + IA ANTI-LOSS ({sistema_ia_status})")
        print("💎 CRITÉRIOS RIGOROSOS + AI LEARNING + PROTEÇÃO ANTI-LOSS!")
    
    def configurar_operacao(self, modo_selecao: str, par_individual: str = None, sniper_only: bool = False):
        """Configura modo de operação"""
        self.config_operacao['modo_selecao'] = modo_selecao
        self.config_operacao['par_individual'] = par_individual
        self.config_operacao['sniper_only'] = sniper_only
        
        if modo_selecao == 'PAR_INDIVIDUAL' and par_individual:
            self.config_operacao['pares_selecionados'] = [par_individual]
        else:
            self.config_operacao['pares_selecionados'] = ['btcusdt', 'ethusdt', 'solusdt', 'xrpusdt', 'adausdt']
    
    def buscar_dados_mercado_v8(self, par: str, timeframe: str, limite: int = 200) -> pd.DataFrame:
        """Busca dados do mercado com cache"""
        
        cache_key = f"{par}_{timeframe}_{int(time.time())//60}"
        if cache_key in self.cache_dados:
            return self.cache_dados[cache_key]
        
        max_tentativas = 3
        for tentativa in range(max_tentativas):
            try:
                tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h'}
                binance_tf = tf_map.get(timeframe, '1m')
                url = f"https://api.binance.com/api/v3/klines?symbol={par.upper()}&interval={binance_tf}&limit={limite}"
                
                response = requests.get(url, timeout=5)  # 🔧 CORREÇÃO: Timeout reduzido de 15s para 5s
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data or len(data) < 10:
                        time.sleep(2)
                        continue
                    
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        if df[col].isna().sum() > len(df) * 0.1:
                            break
                    else:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        if len(df) < 10:
                            time.sleep(2)
                            continue
                        
                        self.cache_dados[cache_key] = df
                        
                        # Limpar cache se muito grande
                        if len(self.cache_dados) > 20:
                            old_keys = list(self.cache_dados.keys())[:5]
                            for key in old_keys:
                                del self.cache_dados[key]
                        
                        return df
                    
                    time.sleep(2)
                    continue
                    
                else:
                    time.sleep(3)
                    
            except Exception as e:
                print(f"⚠️ Erro conexão API (tentativa {tentativa+1}): {e}")
                time.sleep(2 if tentativa < 2 else 5)  # 🔧 CORREÇÃO: Sleep progressivo
        
        return pd.DataFrame()
    
    def verificar_cooldown_inteligente(self, par: str) -> bool:
        """Verifica cooldowns rigorosos"""
        tempo_atual = int(time.time())
        
        # COOLDOWN PÓS-LOSS (RIGOROSO + ADA ESPECIAL)
        if par in self.cooldown_pos_loss:
            cooldown_loss = ConfigRoyalSupremeEnhanced.COOLDOWN_POS_LOSS_ADA if par == 'adausdt' else ConfigRoyalSupremeEnhanced.COOLDOWN_POS_LOSS
            if tempo_atual - self.cooldown_pos_loss[par] < cooldown_loss:
                return False
            else:
                del self.cooldown_pos_loss[par]
        
        # COOLDOWN PÓS-WIN (RIGOROSO)
        if par in self.cooldown_pos_win:
            if tempo_atual - self.cooldown_pos_win[par] < ConfigRoyalSupremeEnhanced.COOLDOWN_POS_WIN:
                return False
            else:
                del self.cooldown_pos_win[par]
        
        if par not in self.cooldown_inteligente:
            return True
        
        # COOLDOWN NORMAL (RIGOROSO)
        cooldown_necessario = ConfigRoyalSupremeEnhanced.COOLDOWN_NORMAL_SEGUNDOS
        
        tempo_decorrido = tempo_atual - self.cooldown_inteligente[par]
        
        if tempo_decorrido >= cooldown_necessario:
            del self.cooldown_inteligente[par]
            return True
        
        return False
    
    def aplicar_auto_calibrador(self, par: str, score_indicadores: float, analise: Dict) -> Tuple[float, bool]:
        """🔧 CORREÇÃO CRÍTICA: Auto calibrador funcional"""
        if not ConfigRoyalSupremeEnhanced.AUTO_CALIBRADOR_ENABLED:
            return score_indicadores, False
        
        try:
            # 🔧 CORREÇÃO: Usar valores locais ao invés do atributo inexistente
            ajustes_por_par = {
                'btcusdt': 0,
                'ethusdt': 5,
                'solusdt': 8,
                'xrpusdt': 10,
                'adausdt': 15
            }
            
            par_adjustment = ajustes_por_par.get(par, 0)
            
            # 🔧 CORREÇÃO: Lógica do calibrador simplificada e funcional
            calibrador_usado = False
            score_final = score_indicadores
            
            # Aplicar ajuste base do par
            if par_adjustment > 0:
                score_final += par_adjustment
                calibrador_usado = True
            
            # Ajuste dinâmico baseado na volatilidade
            volatilidade = analise.get('volatilidade', 0.3)
            if volatilidade > 0.5:  # Alta volatilidade
                score_final += 10
                calibrador_usado = True
            elif volatilidade < 0.15:  # Baixa volatilidade
                score_final += 5
                calibrador_usado = True
            
            # Ajuste baseado no volume
            volume_ratio = analise.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:  # Volume muito alto
                score_final += 15
                calibrador_usado = True
            elif volume_ratio < 0.8:  # Volume baixo - penalizar
                score_final -= 10
                calibrador_usado = True
            
            # Ajuste baseado no horário (funcionalidade original)
            hora_atual = datetime.datetime.now(TIMEZONE).hour
            if 9 <= hora_atual <= 16:  # Horário de maior liquidez
                score_final += 5
                calibrador_usado = True
            elif 2 <= hora_atual <= 6:  # Horário de baixa liquidez - penalizar
                score_final -= 15
                calibrador_usado = True
            
            # Ajuste baseado no enhanced weight
            enhanced_weight = analise.get('enhanced_weight_aplicado', 1.0)
            if enhanced_weight < 0.9:  # Pares com peso menor
                weight_adjustment = (1.0 - enhanced_weight) * 20
                score_final += weight_adjustment
                calibrador_usado = True
            
            # 🔧 CORREÇÃO: Garantir que score não ultrapasse limites
            score_final = max(0, min(100, score_final))
            
            if calibrador_usado:
                self.ia_stats['calibrador_execucoes'] += 1
                print(f"🤖 Auto Calibrador: {par.upper()} {score_indicadores:.0f} → {score_final:.0f}")
            
            return score_final, calibrador_usado
            
        except Exception as e:
            print(f"⚠️ Erro no auto calibrador: {e}")
            return score_indicadores, False
    
    def selecionar_melhor_par(self, analises_pares: Dict[str, Dict]) -> Optional[Tuple[str, Dict]]:
        """Seleciona o melhor par para operar"""
        
        if not analises_pares:
            return None
        
        melhor_par = None
        melhor_score_total = 0
        melhor_analise = None
        
        for par, analise in analises_pares.items():
            if analise.get('analise_bloqueada', False):
                continue
            
            score_call = analise.get('score_call', 0)
            score_put = analise.get('score_put', 0)
            confluencia_call = analise.get('confluencia_call', 0)
            confluencia_put = analise.get('confluencia_put', 0)
            
            enhanced_weight = analise.get('enhanced_weight_aplicado', 1.0)
            volatilidade_base = analise.get('volatilidade_base', 1.0)
            
            if score_call > score_put:
                score_total = score_call + (confluencia_call * 5)
                analise['melhor_direcao'] = 'CALL'
                analise['score_final'] = score_call
                analise['confluencia_final'] = confluencia_call
            else:
                score_total = score_put + (confluencia_put * 5)
                analise['melhor_direcao'] = 'PUT'
                analise['score_final'] = score_put
                analise['confluencia_final'] = confluencia_put
            
            # VOLATILIDADE FILTER RIGOROSO (MANTIDO)
            min_score_ajustado = ConfigRoyalSupremeEnhanced.MIN_SCORE_NORMAL * volatilidade_base
            if analise['score_final'] < min_score_ajustado:
                continue
            
            if score_total > melhor_score_total:
                melhor_score_total = score_total
                melhor_par = par
                melhor_analise = analise
        
        if melhor_par and melhor_analise:
            return (melhor_par, melhor_analise)
        
        return None
    
    def gerar_sinal_royal_supreme_enhanced(self, par: str, timeframe: str) -> Optional[SinalRoyalSupremeEnhanced]:
        """Gera sinal com todos os filtros e correções + IA ANTI-LOSS"""
        try:
            if not self.verificar_cooldown_inteligente(par):
                return None
            
            df = self.buscar_dados_mercado_v8(par, timeframe, 200)
            if df.empty or len(df) < 100:
                return None
            
            analise = self.analisador.analisar_completo_anti_loss_enhanced(df, par)
            
            if analise.get('analise_bloqueada', False):
                return None
            
            score_call = analise['score_call']
            score_put = analise['score_put']
            
            if score_call > score_put:
                melhor_direcao = "CALL"
                score_indicadores = score_call
                motivos = analise['motivos_call']
                confluencia = analise['confluencia_call']
            else:
                melhor_direcao = "PUT"
                score_indicadores = score_put
                motivos = analise['motivos_put']
                confluencia = analise['confluencia_put']
            
            # 🔧 CORREÇÃO CRÍTICA: Aplicar auto calibrador ANTES da validação IA
            score_indicadores_original = score_indicadores
            score_indicadores, auto_calibrador_usado = self.aplicar_auto_calibrador(par, score_indicadores, analise)
            
            # Atualizar analise com dados do calibrador
            analise['auto_calibrador_usado'] = auto_calibrador_usado
            analise['score_original'] = score_indicadores_original
            analise['score_calibrado'] = score_indicadores
            
            # 🤖 IA ANTI-LOSS VALIDATION - INTEGRAÇÃO PRINCIPAL
            if IA_ANTI_LOSS_AVAILABLE:
                try:
                    self.ia_stats['sinais_analisados'] += 1
                    
                    validacao_ia = hook_validar_sinal_antes_emissao(
                        df, analise, par, melhor_direcao, score_indicadores, confluencia
                    )
                    
                    if not validacao_ia['entrada_segura']:
                        print(f"🚫 IA ANTI-LOSS BLOQUEOU: {par.upper()} {melhor_direcao}")
                        for motivo in validacao_ia['motivos_bloqueio']:
                            print(f"   🔴 {motivo}")
                        
                        self.ia_stats['sinais_bloqueados'] += 1
                        return None
                    
                    # Aplicar ajuste de score da IA
                    if validacao_ia['ajuste_score_total'] != 0:
                        score_indicadores += validacao_ia['ajuste_score_total']
                        self.ia_stats['ajustes_aplicados'] += 1
                        print(f"🤖 IA ajustou score: {par.upper()} {validacao_ia['ajuste_score_total']:+.0f}")
                    
                except Exception as e:
                    print(f"⚠️ Erro na validação IA: {e}")
                    # Sistema continua em modo fallback
            
            # AVALIAÇÃO DE SEGURANÇA (CORRIGIDA)
            if self.sistema_sobrevivencia:
                avaliacao_seguranca = self.sistema_sobrevivencia.avaliar_entrada_segura(
                    df, par, melhor_direcao, score_indicadores, confluencia, analise
                )
                
                if not avaliacao_seguranca['entrada_segura'] and not avaliacao_seguranca.get('oportunidade_especial', False):
                    return None
            else:
                # Fallback se sistema sobrevivência não disponível
                avaliacao_seguranca = {
                    'entrada_segura': True,
                    'cenario_detectado': MarketScenarioRoyalSupremeEnhanced.NORMAL,
                    'modo_sobrevivencia': SurvivabilityMode.NORMAL,
                    'auto_calibrador_ativo': auto_calibrador_usado
                }
            
            # DETERMINAR TIPO DE SINAL ENHANCED (CRITÉRIOS RIGOROSOS MANTIDOS)
            cenario_atual = avaliacao_seguranca['cenario_detectado']
            
            # TREND PRIORITY MODE
            if (cenario_atual == MarketScenarioRoyalSupremeEnhanced.TREND_OPPORTUNITY and 
                score_indicadores >= 50 and confluencia >= 5):
                tipo_sinal = TipoSinalRoyalSupremeEnhanced.CALL_TREND_PRIORITY if melhor_direcao == "CALL" else TipoSinalRoyalSupremeEnhanced.PUT_TREND_PRIORITY
                min_score_necessario = 50
                min_confluencia_necessaria = 5
            
            # ELLIOTT MODE
            elif (cenario_atual == MarketScenarioRoyalSupremeEnhanced.ELLIOTT_OPPORTUNITY and 
                  score_indicadores >= 55 and confluencia >= 6):
                tipo_sinal = TipoSinalRoyalSupremeEnhanced.CALL_ELLIOTT if melhor_direcao == "CALL" else TipoSinalRoyalSupremeEnhanced.PUT_ELLIOTT
                min_score_necessario = 55
                min_confluencia_necessaria = 6
            
            # ENHANCED MODE (60-74 score + 6-7 confluência) - RIGOROSO
            elif (60 <= score_indicadores < 75 and 6 <= confluencia <= 7):
                tipo_sinal = TipoSinalRoyalSupremeEnhanced.CALL_ENHANCED if melhor_direcao == "CALL" else TipoSinalRoyalSupremeEnhanced.PUT_ENHANCED
                min_score_necessario = 60
                min_confluencia_necessaria = 6
            
            # SURVIVABILITY MODE (80+ score + 10+ confluência) - RIGOROSO
            elif (score_indicadores >= 80 and confluencia >= 10):
                tipo_sinal = TipoSinalRoyalSupremeEnhanced.CALL_SURVIVABILITY if melhor_direcao == "CALL" else TipoSinalRoyalSupremeEnhanced.PUT_SURVIVABILITY
                min_score_necessario = 80
                min_confluencia_necessaria = 10
            
            # WAVE MODE - RIGOROSO
            elif (cenario_atual == MarketScenarioRoyalSupremeEnhanced.WAVE_OPPORTUNITY and 
                  score_indicadores >= ConfigRoyalSupremeEnhanced.MIN_SCORE_WAVE and 
                  confluencia >= ConfigRoyalSupremeEnhanced.MIN_CONFLUENCIA_WAVE):
                tipo_sinal = TipoSinalRoyalSupremeEnhanced.CALL_WAVE if melhor_direcao == "CALL" else TipoSinalRoyalSupremeEnhanced.PUT_WAVE
                min_score_necessario = ConfigRoyalSupremeEnhanced.MIN_SCORE_WAVE
                min_confluencia_necessaria = ConfigRoyalSupremeEnhanced.MIN_CONFLUENCIA_WAVE
            
            # SNIPER MODE - RIGOROSO
            elif (score_indicadores >= ConfigRoyalSupremeEnhanced.MIN_SCORE_SNIPER and 
                  confluencia >= ConfigRoyalSupremeEnhanced.MIN_CONFLUENCIA_SNIPER):
                tipo_sinal = TipoSinalRoyalSupremeEnhanced.CALL_SNIPER if melhor_direcao == "CALL" else TipoSinalRoyalSupremeEnhanced.PUT_SNIPER
                min_score_necessario = ConfigRoyalSupremeEnhanced.MIN_SCORE_SNIPER
                min_confluencia_necessaria = ConfigRoyalSupremeEnhanced.MIN_CONFLUENCIA_SNIPER
            
            # NORMAL MODE - RIGOROSO
            else:
                tipo_sinal = TipoSinalRoyalSupremeEnhanced.CALL if melhor_direcao == "CALL" else TipoSinalRoyalSupremeEnhanced.PUT
                min_score_necessario = ConfigRoyalSupremeEnhanced.MIN_SCORE_NORMAL
                min_confluencia_necessaria = ConfigRoyalSupremeEnhanced.MIN_CONFLUENCIA
            
            # FILTRO SNIPER ONLY - RIGOROSO
            if self.config_operacao['sniper_only']:
                min_score_necessario = max(min_score_necessario, 65)
                min_confluencia_necessaria = max(min_confluencia_necessaria, 8)
                
                tipos_sniper_allowed = [
                    TipoSinalRoyalSupremeEnhanced.CALL_SNIPER, TipoSinalRoyalSupremeEnhanced.PUT_SNIPER,
                    TipoSinalRoyalSupremeEnhanced.CALL_ENHANCED, TipoSinalRoyalSupremeEnhanced.PUT_ENHANCED,
                    TipoSinalRoyalSupremeEnhanced.CALL_SURVIVABILITY, TipoSinalRoyalSupremeEnhanced.PUT_SURVIVABILITY,
                    TipoSinalRoyalSupremeEnhanced.CALL_TREND_PRIORITY, TipoSinalRoyalSupremeEnhanced.PUT_TREND_PRIORITY,
                    TipoSinalRoyalSupremeEnhanced.CALL_ELLIOTT, TipoSinalRoyalSupremeEnhanced.PUT_ELLIOTT
                ]
                
                if tipo_sinal not in tipos_sniper_allowed:
                    return None
            
            # CRITÉRIOS FINAIS RIGOROSOS
            if score_indicadores < min_score_necessario or confluencia < min_confluencia_necessaria:
                return None
            
            # CRIAR ENHANCED FEATURES LIST
            enhanced_features = []
            if analise.get('support_levels'):
                enhanced_features.append("S/R")
            if analise.get('lta') or analise.get('ltb'):
                enhanced_features.append("LTA/LTB")
            if analise.get('pullback') or analise.get('throwback'):
                enhanced_features.append("Pullback/Throwback")
            if analise.get('elliott_pattern'):
                enhanced_features.append("Elliott")
            if analise.get('price_action_patterns'):
                enhanced_features.append("Price Action")
            if auto_calibrador_usado:
                enhanced_features.append("Auto Calibrador")
            
            tempo_atual = int(time.time())
            agora = datetime.datetime.now(TIMEZONE)
            
            sinal = SinalRoyalSupremeEnhanced(
                tipo_sinal=tipo_sinal,
                par=par,
                timestamp=tempo_atual,
                timeframe=timeframe,
                score_total=score_indicadores,
                score_indicadores=score_indicadores,
                score_filtros=0,
                confluencia_count=confluencia,
                motivos_confluencia=motivos[:15],
                cenario_detectado=cenario_atual,
                modo_sobrevivencia=avaliacao_seguranca.get('modo_sobrevivencia', SurvivabilityMode.NORMAL),
                niveis_sr={},
                volatilidade_atual=analise['volatilidade'],
                volume_ratio=analise['volume_ratio'],
                enhanced_weight_aplicado=analise.get('enhanced_weight_aplicado', 1.0),
                auto_calibrador_usado=auto_calibrador_usado,  # 🔧 CORREÇÃO: Usar valor real
                # ENHANCED PRICE ACTION DATA
                price_action_patterns=analise.get('price_action_patterns', []),
                support_levels=analise.get('support_levels', []),
                resistance_levels=analise.get('resistance_levels', []),
                lta=analise.get('lta'),
                ltb=analise.get('ltb'),
                pullback=analise.get('pullback'),
                throwback=analise.get('throwback'),
                elliott_pattern=analise.get('elliott_pattern'),
                price_action_score_call=analise.get('price_action_score_call', 0.0),
                price_action_score_put=analise.get('price_action_score_put', 0.0),
                enhanced_features=enhanced_features,
                horario_emissao=agora.strftime('%H:%M:%S')
            )
            
            # SALVAR NO DATABASE
            sinal_data = {
                'timestamp': tempo_atual,
                'par': par,
                'tipo': tipo_sinal.value,
                'score': score_indicadores,
                'confluencia': confluencia,
                'cenario': cenario_atual.value,
                'volatilidade': analise['volatilidade'],
                'volume_ratio': analise['volume_ratio'],
                'enhanced_weight': analise.get('enhanced_weight_aplicado', 1.0),
                'auto_calibrador_usado': 1 if auto_calibrador_usado else 0,
                'horario': agora.strftime('%H:%M:%S'),
                'enhanced_features': enhanced_features,
                'motivos': motivos[:10]
            }
            
            self.db_manager.salvar_operacao(sinal_data)
            
            # ATUALIZAR COOLDOWNS
            self.ultimo_sinal_par[par] = tempo_atual
            self.sinais_por_hora[par].append(tempo_atual)
            self.cooldown_inteligente[par] = tempo_atual
            
            # LIMPAR HISTÓRICO DE SINAIS POR HORA
            self.sinais_por_hora[par] = [
                t for t in self.sinais_por_hora[par] 
                if tempo_atual - t < 3600
            ]
            
            return sinal
            
        except Exception as e:
            print(f"❌ Erro gerar sinal {par}: {str(e)}")
            return None
    
    def gerar_sinal_multiplos_pares_v8_enhanced(self, timeframe: str) -> Optional[SinalRoyalSupremeEnhanced]:
        """Gera sinal analisando múltiplos pares"""
        
        pares_para_analisar = self.config_operacao['pares_selecionados']
        analises_pares = {}
        
        for par in pares_para_analisar:
            try:
                df = self.buscar_dados_mercado_v8(par, timeframe, 200)
                if df.empty or len(df) < 100:
                    continue
                
                analise = self.analisador.analisar_completo_anti_loss_enhanced(df, par)
                
                if not analise.get('analise_bloqueada', False):
                    analises_pares[par] = analise
                    
            except Exception as e:
                continue
        
        if self.config_operacao['modo_selecao'] in ['SISTEMA_ESCOLHE', 'TODOS_PARES']:
            resultado = self.selecionar_melhor_par(analises_pares)
            if resultado:
                par_escolhido, analise_escolhida = resultado
                return self.gerar_sinal_royal_supreme_enhanced(par_escolhido, timeframe)
        
        elif self.config_operacao['modo_selecao'] == 'PAR_INDIVIDUAL':
            par_individual = self.config_operacao['par_individual']
            if par_individual and par_individual in analises_pares:
                return self.gerar_sinal_royal_supreme_enhanced(par_individual, timeframe)
        
        return None

    def verificar_resultados_royal_supreme_enhanced(self, sinais_ativos: List[SinalRoyalSupremeEnhanced]) -> List[SinalRoyalSupremeEnhanced]:
        """👑 Sistema WinLoss V8 ORIGINAL - TIMING CORRETO (NÃO MEXER) + IA LEARNING"""
        tempo_atual = int(time.time())
        sinais_para_remover = []
        
        for sinal in sinais_ativos:
            try:
                tempo_decorrido = tempo_atual - sinal.timestamp
                
                # TIMING CORRETO - AGUARDA VELA FINALIZAR COMPLETAMENTE
                verificacao_m1 = 90 <= tempo_decorrido <= 95
                verificacao_gale = 150 <= tempo_decorrido <= 155
                
                # VERIFICAÇÃO M1 - VELA JÁ FINALIZOU
                if (sinal.status == StatusSinalRoyalSupremeEnhanced.ATIVO and 
                    verificacao_m1 and 
                    sinal.vela_resultado_m1 is None):
                    
                    try:
                        url = f"https://api.binance.com/api/v3/klines?symbol={sinal.par.upper()}&interval=1m&limit=3"
                        response = requests.get(url, timeout=3)  # 🔧 CORREÇÃO: Timeout reduzido para 3s
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data and len(data) >= 2:
                                # Pega a vela que JÁ FINALIZOU (penúltima)
                                vela = data[-2]
                                open_price = float(vela[1])
                                close_price = float(vela[4])
                                
                                cor_vela = "GREEN" if close_price > open_price else "RED"
                                sinal.vela_resultado_m1 = cor_vela
                                
                                if 'CALL' in sinal.tipo_sinal.value:
                                    win = (cor_vela == "GREEN")
                                else:
                                    win = (cor_vela == "RED")
                                
                                if win:
                                    sinal.status = StatusSinalRoyalSupremeEnhanced.WIN_M1
                                    print(f"🏆 WIN M1: {PARES_CRYPTO[sinal.par]['nome']}")
                                    self.cooldown_pos_win[sinal.par] = int(time.time())
                                    sinais_para_remover.append(sinal)
                                    self.db_manager.atualizar_resultado_operacao(sinal.timestamp, sinal.par, 'WIN_M1')
                                    
                                    # 🤖 IA LEARNING - REGISTRAR RESULTADO WIN M1
                                    if IA_ANTI_LOSS_AVAILABLE:
                                        try:
                                            hook_registrar_resultado_operacao(sinal.timestamp, sinal.par, 'WIN_M1')
                                        except Exception as e:
                                            pass
                                    
                                else:
                                    sinal.status = StatusSinalRoyalSupremeEnhanced.AGUARDANDO_GALE
                                    print(f"🔄 GALE: {PARES_CRYPTO[sinal.par]['nome']}")
                    except Exception as e:
                        print(f"⚠️ Erro verificação M1: {e}")
                        continue  # 🔧 CORREÇÃO: Continue em vez de pass para não travar
                
                # VERIFICAÇÃO GALE - VELA JÁ FINALIZOU
                elif (sinal.status == StatusSinalRoyalSupremeEnhanced.AGUARDANDO_GALE and 
                      verificacao_gale and 
                      sinal.vela_resultado_gale is None):
                    
                    try:
                        url = f"https://api.binance.com/api/v3/klines?symbol={sinal.par.upper()}&interval=1m&limit=3"
                        response = requests.get(url, timeout=3)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data and len(data) >= 2:
                                # Pega a vela que JÁ FINALIZOU (penúltima)
                                vela = data[-2]
                                open_price = float(vela[1])
                                close_price = float(vela[4])
                                
                                cor_vela = "GREEN" if close_price > open_price else "RED"
                                sinal.vela_resultado_gale = cor_vela
                                
                                if 'CALL' in sinal.tipo_sinal.value:
                                    win = (cor_vela == "GREEN")
                                else:
                                    win = (cor_vela == "RED")
                                
                                if win:
                                    sinal.status = StatusSinalRoyalSupremeEnhanced.WIN_GALE
                                    print(f"🏆 WIN GALE: {PARES_CRYPTO[sinal.par]['nome']}")
                                    self.cooldown_pos_win[sinal.par] = int(time.time())
                                    self.db_manager.atualizar_resultado_operacao(sinal.timestamp, sinal.par, 'WIN_GALE')
                                    
                                    # 🤖 IA LEARNING - REGISTRAR RESULTADO WIN GALE
                                    if IA_ANTI_LOSS_AVAILABLE:
                                        try:
                                            hook_registrar_resultado_operacao(sinal.timestamp, sinal.par, 'WIN_GALE')
                                        except Exception as e:
                                            pass
                                    
                                else:
                                    sinal.status = StatusSinalRoyalSupremeEnhanced.LOSS
                                    print(f"💎 LOSS: {PARES_CRYPTO[sinal.par]['nome']}")
                                    print(f"👑 'Royal Supreme Enhanced + AI Never Gives Up!'")
                                    self.cooldown_pos_loss[sinal.par] = int(time.time())
                                    self.db_manager.atualizar_resultado_operacao(sinal.timestamp, sinal.par, 'LOSS')
                                    
                                    # 🤖 IA LEARNING - REGISTRAR RESULTADO LOSS
                                    if IA_ANTI_LOSS_AVAILABLE:
                                        try:
                                            hook_registrar_resultado_operacao(sinal.timestamp, sinal.par, 'LOSS')
                                        except Exception as e:
                                            pass
                                
                                sinais_para_remover.append(sinal)
                    except Exception as e:
                        print(f"⚠️ Erro verificação GALE: {e}")
                        continue  # 🔧 CORREÇÃO: Continue em vez de pass para não travar
                
                # Timeout
                elif tempo_decorrido > ConfigRoyalSupremeEnhanced.TIMEOUT_SINAL_SEGUNDOS:
                    print(f"⏰ TIMEOUT: {PARES_CRYPTO[sinal.par]['nome']}")
                    sinal.status = StatusSinalRoyalSupremeEnhanced.TIMEOUT
                    self.db_manager.atualizar_resultado_operacao(sinal.timestamp, sinal.par, 'TIMEOUT')
                    
                    # 🤖 IA LEARNING - REGISTRAR TIMEOUT
                    if IA_ANTI_LOSS_AVAILABLE:
                        try:
                            hook_registrar_resultado_operacao(sinal.timestamp, sinal.par, 'TIMEOUT')
                        except Exception as e:
                            pass
                    
                    sinais_para_remover.append(sinal)
            
            except Exception as e:
                print(f"⚠️ Erro verificando sinal {sinal.par}: {e}")
                continue  # 🔧 CORREÇÃO: Continue para não travar loop principal
        
        return sinais_para_remover
    
    def get_stats_ia_anti_loss(self) -> Dict[str, Any]:
        """Retorna estatísticas da IA Anti-Loss"""
        if not IA_ANTI_LOSS_AVAILABLE:
            return {
                'sistema_ativo': False,
                'motivo': 'IA Anti-Loss não disponível'
            }
        
        try:
            from integration_hooks import hook_obter_stats_ia
            stats_completas = hook_obter_stats_ia()
            
            # Adicionar stats locais
            stats_completas['stats_engine'] = self.ia_stats
            
            # Calcular métricas locais
            total = self.ia_stats['sinais_analisados']
            if total > 0:
                stats_completas['metricas_engine'] = {
                    'taxa_bloqueio_local': (self.ia_stats['sinais_bloqueados'] / total * 100),
                    'taxa_ajustes_local': (self.ia_stats['ajustes_aplicados'] / total * 100),
                    'calibrador_uso': (self.ia_stats['calibrador_execucoes'] / total * 100)
                }
            
            return stats_completas
            
        except Exception as e:
            return {
                'sistema_ativo': False,
                'erro': str(e),
                'stats_locais': self.ia_stats
            }
    
    def status_ia_anti_loss(self) -> Dict[str, Any]:
        """Status rápido da IA Anti-Loss"""
        if not IA_ANTI_LOSS_AVAILABLE:
            return {
                'ativo': False,
                'motivo': 'IA não carregada',
                'componentes': 0,
                'sinais_analisados': 0,
                'calibrador_execucoes': 0
            }
        
        try:
            from integration_hooks import hook_status_sistema_ia
            status = hook_status_sistema_ia()
            
            # Adicionar dados locais
            status['engine_stats'] = self.ia_stats
            
            return status
        except Exception as e:
            return {
                'ativo': False,
                'erro': str(e),
                'engine_stats': self.ia_stats
            }

print("✅ ENGINE ROYAL SUPREME ENHANCED + DATABASE AI + IA ANTI-LOSS CARREGADO!")
print("🤖 INTEGRAÇÃO IA ANTI-LOSS COMPLETA - PROTEÇÃO MÁXIMA ATIVADA!")
print("🔧 AUTO CALIBRADOR FUNCIONAL - SISTEMA CORRIGIDO!")