#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 CIPHER ROYAL SUPREME ENHANCED + DATABASE AI - MAIN LAUNCHER 👑
💎 SISTEMA PRINCIPAL QUE IMPORTA TODOS OS MÓDULOS
🔥 EXECUTE ESTE ARQUIVO PARA RODAR O SISTEMA COMPLETO
"""

import os
import sys
import time
import datetime
import pytz

# 🔧 CORREÇÃO: Try/catch em todos os imports para evitar erros (LOGS OTIMIZADOS)
print("💎 Carregando módulos do sistema...")

try:
    from config_royal import *
except ImportError as e:
    print(f"❌ Erro crítico ao carregar config_royal: {e}")
    sys.exit(1)

try:
    from database_manager import DatabaseManager
except ImportError as e:
    print(f"❌ Erro crítico ao carregar database_manager: {e}")
    sys.exit(1)

try:
    from enhanced_technical import EnhancedTechnicalAnalysis, PriceActionMasterRoyalSupremeEnhanced
except ImportError as e:
    print(f"❌ Erro crítico ao carregar enhanced_technical: {e}")
    sys.exit(1)

try:
    from arsenal_tecnico import ArsenalTecnicoCompletoV8RoyalSupremeEnhanced
except ImportError as e:
    print(f"❌ Erro crítico ao carregar arsenal_tecnico: {e}")
    sys.exit(1)

try:
    from detectores_mercado import DetectorMercadoCaoticoV8RoyalSupremeEnhanced, DetectorCenariosExtremosV8RoyalSupremeEnhanced
except ImportError as e:
    print(f"❌ Erro crítico ao carregar detectores_mercado: {e}")
    sys.exit(1)

try:
    from telegram_system import RoyalSessionManagerSupremeEnhanced, TelegramAdminCommandsEnhanced, RoyalTelegramSupremeEnhanced
except ImportError as e:
    print(f"❌ Erro crítico ao carregar telegram_system: {e}")
    sys.exit(1)

# 🔧 CORREÇÃO CRÍTICA: Importar analisador com tratamento de erro
try:
    from analisador_completo import (
        SistemaSobrevivenciaV8RoyalSupremeEnhanced, 
        AnalisadorCompletoV8RoyalSupremeEnhanced, 
        SinalRoyalSupremeEnhanced, 
        RelatoriosRoyalSupremeEnhanced
    )
except ImportError as e:
    print(f"❌ Erro crítico ao carregar analisador_completo: {e}")
    print("🚨 SUBSTITUA O ARQUIVO analisador_completo.py PELO CORRIGIDO!")
    sys.exit(1)

try:
    from engine_royal import EngineRoyalSupremeEnhanced
except ImportError as e:
    print(f"❌ Erro crítico ao carregar engine_royal: {e}")
    sys.exit(1)

try:
    from cipher_system import CipherRoyalSupremeEnhanced
except ImportError as e:
    print(f"❌ Erro crítico ao carregar cipher_system: {e}")
    sys.exit(1)

# 📰 NOVO: Carregar News Analyzer (opcional - SEM LOG REPETITIVO)
try:
    from news_analyzer import NewsImpactAnalyzer
    NEWS_SYSTEM_AVAILABLE = True
except ImportError:
    NEWS_SYSTEM_AVAILABLE = False

# ✅ LOG ÚNICO DE SUCESSO
print("✅ Todos os módulos carregados com sucesso!")

print('\033[1;38;5;196m' + """
👑━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━👑
█                                                                              █
█  💎 CIPHER ROYAL SUPREME ENHANCED + DATABASE AI - MAIN LAUNCHER 💎         █
█                                                                              █
█              👑 RIGOROSO + INTELIGENTE + AI LEARNING 👑                    █
█                                                                              █
█    🏆 Sistema Modular: MAIN + Módulos Separados para Performance          █
█                                                                              █
█  ┌────────────────────────────────────────────────────────────────────────┐ █
█  │ 💎 CRITÉRIOS: Royal Supreme rigorosos FUNCIONAIS                      │ █
█  │ ⚡ FEATURES: S/R + LTA/LTB + Pullback + Elliott + Database            │ █
█  │ 🛡️ WIN/LOSS: Sistema correto + Aprendizado automático                │ █
█  │ 👑 RESULTADO: Máxima precisão + Inteligência evolutiva                │ █
█  │ 📰 NEWS: Sistema de notícias RSS integrado                            │ █
█  └────────────────────────────────────────────────────────────────────────┘ █
█                                                                              █
█           🏆 ROYAL SUPREME ENHANCED WITH AI NEVER LOSES 🏆                 █
█                                                                              █
👑━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━👑
""" + '\033[0m\n')

# Verificar se todos os módulos existem
MODULOS_NECESSARIOS = [
    'config_royal',
    'database_manager', 
    'enhanced_technical',
    'arsenal_tecnico',
    'detectores_mercado',
    'telegram_system',
    'analisador_completo',
    'engine_royal',
    'cipher_system'
]

MODULOS_OPCIONAIS = [
    'news_analyzer'
]

def verificar_modulos():
    """Verifica se todos os módulos necessários existem"""
    modulos_faltando = []
    modulos_opcionais_faltando = []
    
    # Verificar módulos obrigatórios
    for modulo in MODULOS_NECESSARIOS:
        try:
            if not os.path.exists(f"{modulo}.py"):
                modulos_faltando.append(f"{modulo}.py")
        except:
            modulos_faltando.append(f"{modulo}.py")
    
    # Verificar módulos opcionais
    for modulo in MODULOS_OPCIONAIS:
        try:
            if not os.path.exists(f"{modulo}.py"):
                modulos_opcionais_faltando.append(f"{modulo}.py")
        except:
            modulos_opcionais_faltando.append(f"{modulo}.py")
    
    if modulos_faltando:
        print("❌ MÓDULOS OBRIGATÓRIOS FALTANDO:")
        for modulo in modulos_faltando:
            print(f"   🔍 {modulo}")
        print("\n🚨 CRIE OS MÓDULOS FALTANTES PRIMEIRO!")
        return False
    
    # 🔧 CORREÇÃO: Não mostrar módulos opcionais faltando se só for news_analyzer
    # (evita spam de logs)
    
    return True

def main():
    """Função principal do sistema"""
    try:
        print("👑 CIPHER ROYAL SUPREME ENHANCED + DATABASE AI - MAIN LAUNCHER")
        print("💎 VERIFICANDO SISTEMA...")
        
        # Verificar módulos
        if not verificar_modulos():
            print("\n🛑 Sistema não pode ser iniciado - módulos obrigatórios faltando")
            input("Pressione Enter para sair...")
            return
        
        print("🏆 SISTEMA VERIFICADO E PRONTO!")
        
        # 📰 Status do News System (LOG ÚNICO)
        if NEWS_SYSTEM_AVAILABLE:
            print("📰 News System: ATIVO")
        else:
            print("📰 News System: DESATIVADO")
        
        print("💎 INICIANDO CIPHER ROYAL SUPREME ENHANCED + DATABASE AI")
        print("🔥 'ROYAL SUPREME ENHANCED + AI NEVER LOSES!'\n")
        
        # Criar e iniciar sistema
        royal_supreme_enhanced = CipherRoyalSupremeEnhanced()
        
        config = royal_supreme_enhanced.iniciar_sistema_automatico()
        
        if config is None:
            print("🛑 Sistema cancelado pelo usuário")
            return
        
        # Executar sistema principal
        royal_supreme_enhanced.executar_ciclo_royal_supreme_enhanced(config)
        
    except KeyboardInterrupt:
        print(f"\n🛑 Royal Supreme Enhanced + AI System interrupted")
    except Exception as e:
        print(f"\n❌ CRITICAL ROYAL SUPREME ENHANCED + AI ERROR: {e}")
        print("🔧 Detalhes do erro:")
        import traceback
        traceback.print_exc()
        
        print("\n🛠️ POSSÍVEIS SOLUÇÕES:")
        print("1. Verifique se todos os arquivos .py existem")
        print("2. Substitua analisador_completo.py pela versão corrigida")
        print("3. Verifique se config_royal.py tem todas as configurações")
        print("4. Reinicie o sistema")

if __name__ == "__main__":
    main()

print("\n👑 MAIN LAUNCHER FINALIZADO!")
print("🔥 'ROYAL SUPREME ENHANCED + DATABASE AI MODULAR SYSTEM!' 💎🗄️🤖")