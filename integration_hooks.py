#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔗 INTEGRATION HOOKS - CIPHER ROYAL SUPREME ENHANCED + AI ANTI-LOSS 🔗
💎 HOOKS DE INTEGRAÇÃO PARA O SISTEMA PRINCIPAL
🔥 INTEGRAÇÃO MODULAR SEM QUEBRAR O SISTEMA EXISTENTE
🎯 APENAS 3-5 LINHAS ADICIONADAS AO SISTEMA ORIGINAL
"""

from typing import Dict, Any, Optional
import time

# Importar todos os módulos IA
try:
    from ai_predictor import AIValidatorIntegration
    AI_PREDICTOR_AVAILABLE = True
except ImportError:
    AI_PREDICTOR_AVAILABLE = False
    print("⚠️ AI Predictor não disponível - sistema funcionará sem IA")

try:
    from filtros_contexto import FiltrosIntegration
    FILTROS_AVAILABLE = True
except ImportError:
    FILTROS_AVAILABLE = False
    print("⚠️ Filtros Contexto não disponíveis")

try:
    from detector_cenarios_perigosos import DetectoresIntegration
    DETECTORES_AVAILABLE = True
except ImportError:
    DETECTORES_AVAILABLE = False
    print("⚠️ Detectores Cenários não disponíveis")

try:
    from dataset_collector import DatasetCollectorIntegration
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("⚠️ Dataset Collector não disponível")

class AIAntiLossSystem:
    """🤖 SISTEMA INTEGRADO IA ANTI-LOSS"""
    
    def __init__(self):
        self.enabled = False
        self.ai_validator = None
        self.filtros = None
        self.detectores = None
        self.dataset_collector = None
        
        # Configurações
        self.config = {
            'usar_ia_predictor': True,
            'usar_filtros_contexto': True,
            'usar_detectores_cenarios': True,
            'usar_dataset_collector': True,
            'modo_ia': 'moderate',  # conservative, moderate, aggressive
            'modo_fallback': True,  # Se IA falhar, permitir sinal
            'log_detalhado': True
        }
        
        # Estatísticas
        self.stats = {
            'sinais_analisados': 0,
            'sinais_bloqueados_ia': 0,
            'sinais_bloqueados_filtros': 0,
            'sinais_bloqueados_detectores': 0,
            'ajustes_score_aplicados': 0,
            'tempo_total_analise_ms': 0
        }
        
        self.inicializar_componentes()
    
    def inicializar_componentes(self):
        """Inicializa componentes disponíveis"""
        try:
            # AI Predictor
            if AI_PREDICTOR_AVAILABLE and self.config['usar_ia_predictor']:
                self.ai_validator = AIValidatorIntegration()
                print("✅ AI Predictor integrado")
            
            # Filtros Contexto
            if FILTROS_AVAILABLE and self.config['usar_filtros_contexto']:
                self.filtros = FiltrosIntegration()
                print("✅ Filtros Contexto integrados")
            
            # Detectores Cenários
            if DETECTORES_AVAILABLE and self.config['usar_detectores_cenarios']:
                self.detectores = DetectoresIntegration()
                print("✅ Detectores Cenários integrados")
            
            # Dataset Collector
            if DATASET_AVAILABLE and self.config['usar_dataset_collector']:
                self.dataset_collector = DatasetCollectorIntegration()
                print("✅ Dataset Collector integrado")
            
            self.enabled = True
            print("🚀 SISTEMA IA ANTI-LOSS INICIALIZADO COM SUCESSO!")
            
        except Exception as e:
            print(f"⚠️ Erro inicializando IA Anti-Loss: {e}")
            self.enabled = False
    
    def validar_entrada_completa(self, df, analise_completa: Dict, par: str, 
                                tipo_sinal: str, score_total: float, 
                                confluencia_count: int) -> Dict[str, Any]:
        """
        🎯 FUNÇÃO PRINCIPAL DE VALIDAÇÃO
        Esta é a única função que o sistema principal precisa chamar
        """
        inicio_tempo = time.time()
        
        if not self.enabled:
            return self._resultado_default(True, "Sistema IA desabilitado")
        
        # Verificação básica de dados
        if not isinstance(analise_completa, dict) or not par or not tipo_sinal:
            return self._resultado_default(True, "Dados inválidos para IA")
        
        self.stats['sinais_analisados'] += 1
        
        try:
            resultado_final = {
                'entrada_segura': True,
                'motivos_bloqueio': [],
                'ajuste_score_total': 0,
                'confianca_geral': 0.5,
                'detalhes_ia': {},
                'sistema_ativo': True
            }
            
            # 1. VALIDAÇÃO IA PREDICTOR
            if self.ai_validator:
                try:
                    resultado_ia = self.ai_validator.validar_entrada(
                        df, analise_completa, par, tipo_sinal, score_total, confluencia_count
                    )
                    
                    if not resultado_ia['entrada_segura']:
                        resultado_final['entrada_segura'] = False
                        resultado_final['motivos_bloqueio'].append(f"IA: {resultado_ia['motivo_bloqueio']}")
                        self.stats['sinais_bloqueados_ia'] += 1
                    
                    if abs(resultado_ia['score_adjustment']) > 0:
                        resultado_final['ajuste_score_total'] += resultado_ia['score_adjustment']
                        self.stats['ajustes_score_aplicados'] += 1
                    
                    resultado_final['detalhes_ia']['ai_predictor'] = resultado_ia
                    resultado_final['confianca_geral'] = resultado_ia.get('ia_confidence', 0.5)
                    
                except Exception as e:
                    if not self.config['modo_fallback']:
                        resultado_final['entrada_segura'] = False
                        resultado_final['motivos_bloqueio'].append(f"Erro IA: {str(e)}")
                    
                    if self.config['log_detalhado']:
                        print(f"⚠️ Erro AI Predictor: {e}")
            
            # 2. VALIDAÇÃO FILTROS CONTEXTO
            if self.filtros:
                try:
                    resultado_filtros = self.filtros.validar_contexto(
                        df, analise_completa, par, tipo_sinal, score_total
                    )
                    
                    if not resultado_filtros['entrada_segura']:
                        resultado_final['entrada_segura'] = False
                        resultado_final['motivos_bloqueio'].extend(resultado_filtros['motivos_bloqueio'])
                        self.stats['sinais_bloqueados_filtros'] += 1
                    
                    if resultado_filtros['ajuste_score'] != 0:
                        resultado_final['ajuste_score_total'] += resultado_filtros['ajuste_score']
                    
                    resultado_final['detalhes_ia']['filtros_contexto'] = resultado_filtros
                    
                except Exception as e:
                    if self.config['log_detalhado']:
                        print(f"⚠️ Erro Filtros: {e}")
            
            # 3. VALIDAÇÃO DETECTORES CENÁRIOS
            if self.detectores:
                try:
                    resultado_detectores = self.detectores.validar_cenarios(
                        df, analise_completa, par, tipo_sinal, score_total
                    )
                    
                    if not resultado_detectores['entrada_segura']:
                        resultado_final['entrada_segura'] = False
                        resultado_final['motivos_bloqueio'].extend(resultado_detectores['motivos_bloqueio'])
                        self.stats['sinais_bloqueados_detectores'] += 1
                    
                    resultado_final['detalhes_ia']['detectores_cenarios'] = resultado_detectores
                    
                except Exception as e:
                    if self.config['log_detalhado']:
                        print(f"⚠️ Erro Detectores: {e}")
            
            # 4. LOG DETALHADO
            if self.config['log_detalhado'] and not resultado_final['entrada_segura']:
                print(f"🚫 IA ANTI-LOSS BLOQUEOU: {par.upper()} {tipo_sinal}")
                for motivo in resultado_final['motivos_bloqueio']:
                    print(f"   🔴 {motivo}")
            
            # 5. REGISTRAR NO DATASET (se disponível)
            if self.dataset_collector and resultado_final['entrada_segura']:
                try:
                    sinal_data = {
                        'timestamp': int(time.time()),
                        'par': par,
                        'tipo': tipo_sinal,
                        'score': score_total,
                        'confluencia': confluencia_count
                    }
                    self.dataset_collector.registrar_sinal_emitido(sinal_data)
                except Exception as e:
                    if self.config['log_detalhado']:
                        print(f"⚠️ Erro Dataset Collector: {e}")
            
            # Atualizar tempo de análise
            tempo_decorrido = (time.time() - inicio_tempo) * 1000
            self.stats['tempo_total_analise_ms'] += tempo_decorrido
            
            return resultado_final
            
        except Exception as e:
            print(f"❌ Erro crítico IA Anti-Loss: {e}")
            return self._resultado_default(self.config['modo_fallback'], f"Erro crítico: {str(e)}")
    
    def registrar_resultado_operacao(self, timestamp: int, par: str, resultado: str):
        """Registra resultado de operação para aprendizado"""
        if self.dataset_collector:
            try:
                self.dataset_collector.registrar_resultado_sinal(timestamp, par, resultado)
            except Exception as e:
                if self.config['log_detalhado']:
                    print(f"⚠️ Erro registrando resultado: {e}")
    
    def _resultado_default(self, entrada_segura: bool, motivo: str = "") -> Dict[str, Any]:
        """Resultado padrão quando sistema não está disponível"""
        return {
            'entrada_segura': entrada_segura,
            'motivos_bloqueio': [motivo] if motivo else [],
            'ajuste_score_total': 0,
            'confianca_geral': 0.5,
            'detalhes_ia': {},
            'sistema_ativo': False
        }
    
    def configurar_sistema(self, **kwargs):
        """Configura o sistema IA"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"🤖 IA configurado: {key} = {value}")
        
        # Reconfigurar componentes se necessário
        if 'modo_ia' in kwargs and self.ai_validator:
            self.ai_validator.configurar_modo(kwargs['modo_ia'])
    
    def get_stats_completas(self) -> Dict[str, Any]:
        """Retorna estatísticas completas do sistema"""
        stats_completas = {
            'sistema_ativo': self.enabled,
            'componentes_ativos': {
                'ai_predictor': self.ai_validator is not None,
                'filtros_contexto': self.filtros is not None,
                'detectores_cenarios': self.detectores is not None,
                'dataset_collector': self.dataset_collector is not None
            },
            'stats_gerais': self.stats.copy()
        }
        
        # Adicionar stats específicos de cada componente
        if self.ai_validator:
            stats_completas['stats_ai_predictor'] = self.ai_validator.get_stats()
        
        if self.filtros:
            stats_completas['stats_filtros'] = self.filtros.get_stats()
        
        if self.detectores:
            stats_completas['stats_detectores'] = self.detectores.get_stats()
        
        if self.dataset_collector:
            stats_completas['stats_dataset'] = self.dataset_collector.get_stats()
        
        # Calcular métricas derivadas
        total_analisados = self.stats['sinais_analisados']
        if total_analisados > 0:
            stats_completas['metricas_derivadas'] = {
                'taxa_bloqueio_ia': (self.stats['sinais_bloqueados_ia'] / total_analisados * 100),
                'taxa_bloqueio_filtros': (self.stats['sinais_bloqueados_filtros'] / total_analisados * 100),
                'taxa_bloqueio_detectores': (self.stats['sinais_bloqueados_detectores'] / total_analisados * 100),
                'tempo_medio_analise_ms': (self.stats['tempo_total_analise_ms'] / total_analisados)
            }
        
        return stats_completas
    
    def gerar_relatorio_completo(self) -> str:
        """Gera relatório completo do sistema IA"""
        stats = self.get_stats_completas()
        
        relatorio = f"""
🤖 RELATÓRIO SISTEMA IA ANTI-LOSS COMPLETO

⚡ STATUS SISTEMA:
   Sistema Ativo: {"✅ SIM" if stats['sistema_ativo'] else "❌ NÃO"}
   
🧩 COMPONENTES:
   AI Predictor: {"✅ ATIVO" if stats['componentes_ativos']['ai_predictor'] else "❌ INATIVO"}
   Filtros Contexto: {"✅ ATIVO" if stats['componentes_ativos']['filtros_contexto'] else "❌ INATIVO"}
   Detectores Cenários: {"✅ ATIVO" if stats['componentes_ativos']['detectores_cenarios'] else "❌ INATIVO"}
   Dataset Collector: {"✅ ATIVO" if stats['componentes_ativos']['dataset_collector'] else "❌ INATIVO"}

📊 ESTATÍSTICAS GERAIS:
   Sinais Analisados: {stats['stats_gerais']['sinais_analisados']}
   Bloqueios IA: {stats['stats_gerais']['sinais_bloqueados_ia']}
   Bloqueios Filtros: {stats['stats_gerais']['sinais_bloqueados_filtros']}
   Bloqueios Detectores: {stats['stats_gerais']['sinais_bloqueados_detectores']}
   Ajustes Score: {stats['stats_gerais']['ajustes_score_aplicados']}
"""
        
        if 'metricas_derivadas' in stats:
            metricas = stats['metricas_derivadas']
            relatorio += f"""
📈 MÉTRICAS DE PERFORMANCE:
   Taxa Bloqueio IA: {metricas['taxa_bloqueio_ia']:.1f}%
   Taxa Bloqueio Filtros: {metricas['taxa_bloqueio_filtros']:.1f}%
   Taxa Bloqueio Detectores: {metricas['taxa_bloqueio_detectores']:.1f}%
   Tempo Médio Análise: {metricas['tempo_medio_analise_ms']:.1f}ms
"""
        
        # Adicionar relatórios específicos
        if self.filtros:
            relatorio += "\n" + self.filtros.gerar_relatorio()
        
        if self.detectores:
            relatorio += "\n" + self.detectores.gerar_relatorio()
        
        if self.dataset_collector:
            relatorio += "\n" + self.dataset_collector.gerar_relatorio()
        
        relatorio += "\n🤖 SISTEMA IA ANTI-LOSS PROTEGENDO CONTRA LOSSES!"
        
        return relatorio

# INSTÂNCIA GLOBAL DO SISTEMA IA
# Esta é a única variável que o sistema principal precisa importar
AI_ANTI_LOSS_SYSTEM = AIAntiLossSystem()

# HOOKS DE INTEGRAÇÃO PRINCIPAIS
def hook_validar_sinal_antes_emissao(df, analise_completa: Dict, par: str, 
                                    tipo_sinal: str, score_total: float, 
                                    confluencia_count: int) -> Dict[str, Any]:
    """
    🎯 HOOK PRINCIPAL: Chamar esta função antes de emitir qualquer sinal
    
    INTEGRAÇÃO NO SISTEMA PRINCIPAL:
    No método gerar_sinal_royal_supreme_enhanced(), adicionar antes do return:
    
    # HOOK IA ANTI-LOSS
    from integration_hooks import hook_validar_sinal_antes_emissao
    validacao_ia = hook_validar_sinal_antes_emissao(df, analise, par, tipo_sinal.value, score_indicadores, confluencia)
    if not validacao_ia['entrada_segura']:
        return None  # Bloquear sinal
    score_indicadores += validacao_ia['ajuste_score_total']  # Aplicar ajuste
    """
    return AI_ANTI_LOSS_SYSTEM.validar_entrada_completa(
        df, analise_completa, par, tipo_sinal, score_total, confluencia_count
    )

def hook_registrar_resultado_operacao(timestamp: int, par: str, resultado: str):
    """
    📊 HOOK RESULTADO: Chamar quando resultado da operação for conhecido
    
    INTEGRAÇÃO NO SISTEMA PRINCIPAL:
    No método verificar_resultados_royal_supreme_enhanced(), adicionar:
    
    # HOOK IA RESULTADO
    from integration_hooks import hook_registrar_resultado_operacao
    hook_registrar_resultado_operacao(sinal.timestamp, sinal.par, resultado_string)
    """
    AI_ANTI_LOSS_SYSTEM.registrar_resultado_operacao(timestamp, par, resultado)

def hook_configurar_ia_sistema(**kwargs):
    """
    ⚙️ HOOK CONFIGURAÇÃO: Configurar sistema IA
    
    Exemplos de uso:
    hook_configurar_ia_sistema(modo_ia='conservative')
    hook_configurar_ia_sistema(usar_ia_predictor=False)
    """
    AI_ANTI_LOSS_SYSTEM.configurar_sistema(**kwargs)

def hook_obter_stats_ia() -> Dict[str, Any]:
    """📈 HOOK STATS: Obter estatísticas do sistema IA"""
    return AI_ANTI_LOSS_SYSTEM.get_stats_completas()

def hook_gerar_relatorio_ia() -> str:
    """📋 HOOK RELATÓRIO: Gerar relatório completo"""
    return AI_ANTI_LOSS_SYSTEM.gerar_relatorio_completo()

def hook_status_sistema_ia() -> Dict[str, Any]:
    """🔍 HOOK STATUS: Status rápido do sistema"""
    return {
        'ativo': AI_ANTI_LOSS_SYSTEM.enabled,
        'componentes': len([c for c in [
            AI_ANTI_LOSS_SYSTEM.ai_validator,
            AI_ANTI_LOSS_SYSTEM.filtros,
            AI_ANTI_LOSS_SYSTEM.detectores,
            AI_ANTI_LOSS_SYSTEM.dataset_collector
        ] if c is not None]),
        'sinais_analisados': AI_ANTI_LOSS_SYSTEM.stats['sinais_analisados']
    }

# EXEMPLO DE INTEGRAÇÃO MÍNIMA PARA O SISTEMA PRINCIPAL
"""
INTEGRAÇÃO MÍNIMA NECESSÁRIA:

1. No arquivo engine_royal.py, no método gerar_sinal_royal_supreme_enhanced():

# Adicionar no início do arquivo:
try:
    from integration_hooks import hook_validar_sinal_antes_emissao, hook_registrar_resultado_operacao
    IA_ANTI_LOSS_AVAILABLE = True
except ImportError:
    IA_ANTI_LOSS_AVAILABLE = False

# Adicionar antes do return sinal (linha ~380):
if IA_ANTI_LOSS_AVAILABLE:
    validacao_ia = hook_validar_sinal_antes_emissao(df, analise, par, tipo_sinal.value, score_indicadores, confluencia)
    if not validacao_ia['entrada_segura']:
        return None
    score_indicadores += validacao_ia['ajuste_score_total']

2. No método verificar_resultados_royal_supreme_enhanced():

# Adicionar quando resultado for definido:
if IA_ANTI_LOSS_AVAILABLE:
    hook_registrar_resultado_operacao(sinal.timestamp, sinal.par, sinal.status.value)

3. No arquivo cipher_system.py, adicionar comando admin:

# Adicionar no display ou em método separado:
def mostrar_stats_ia():
    if IA_ANTI_LOSS_AVAILABLE:
        from integration_hooks import hook_gerar_relatorio_ia
        print(hook_gerar_relatorio_ia())

APENAS ESSAS 5-7 LINHAS SÃO NECESSÁRIAS PARA INTEGRAÇÃO COMPLETA!
"""

print("✅ INTEGRATION HOOKS CARREGADOS - READY FOR SEAMLESS INTEGRATION!")