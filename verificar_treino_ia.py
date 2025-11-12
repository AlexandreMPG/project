#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔍 VERIFICADOR DE TREINO DA IA - CIPHER ROYAL SUPREME
📊 Verifica se os modelos foram treinados e salvos corretamente
"""

import os
import pickle
import sqlite3
import time
from datetime import datetime

def verificar_treino_ia():
    print("🔍 VERIFICANDO TREINO DA IA ANTI-LOSS")
    print("="*50)
    
    # 1. Verificar se pasta ai_models existe
    pasta_modelos = 'ai_models/'
    if not os.path.exists(pasta_modelos):
        print("❌ Pasta ai_models/ não encontrada!")
        print("💡 Execute: python ai_model_trainer.py")
        return False
    
    print("✅ Pasta ai_models/ encontrada")
    
    # 2. Verificar arquivos dos modelos
    arquivos_necessarios = [
        'xgboost_model.pkl',
        'random_forest_model.pkl', 
        'scaler.pkl',
        'metadata.pkl'
    ]
    
    arquivos_encontrados = []
    for arquivo in arquivos_necessarios:
        caminho = os.path.join(pasta_modelos, arquivo)
        if os.path.exists(caminho):
            tamanho = os.path.getsize(caminho)
            print(f"✅ {arquivo} - {tamanho} bytes")
            arquivos_encontrados.append(arquivo)
        else:
            print(f"❌ {arquivo} - NÃO ENCONTRADO")
    
    if len(arquivos_encontrados) != len(arquivos_necessarios):
        print(f"\n⚠️ Apenas {len(arquivos_encontrados)}/{len(arquivos_necessarios)} arquivos encontrados")
        return False
    
    # 3. Verificar conteúdo dos modelos
    try:
        print("\n📊 VERIFICANDO CONTEÚDO DOS MODELOS:")
        
        # Carregar metadata
        with open(os.path.join(pasta_modelos, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        timestamp_treino = metadata.get('timestamp', 0)
        data_treino = datetime.fromtimestamp(timestamp_treino).strftime('%d/%m/%Y %H:%M:%S')
        print(f"📅 Data do treino: {data_treino}")
        
        # Verificar métricas dos modelos
        model_metrics = metadata.get('model_metrics', {})
        
        for modelo, metricas in model_metrics.items():
            print(f"\n🤖 {modelo.upper()}:")
            print(f"   Accuracy: {metricas.get('accuracy', 0):.3f}")
            print(f"   Cross-Validation: {metricas.get('cv_mean', 0):.3f} ± {metricas.get('cv_std', 0):.3f}")
        
        # Verificar feature importance
        feature_importance = metadata.get('feature_importance', {})
        if feature_importance:
            print(f"\n🎯 TOP 5 FEATURES MAIS IMPORTANTES:")
            for modelo, features in feature_importance.items():
                top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\n   {modelo.upper()}:")
                for feature, importance in top_features:
                    print(f"     {feature}: {importance:.3f}")
        
        # Verificar se modelos podem ser carregados
        print(f"\n🔧 TESTANDO CARREGAMENTO DOS MODELOS:")
        
        with open(os.path.join(pasta_modelos, 'xgboost_model.pkl'), 'rb') as f:
            xgb_model = pickle.load(f)
            print(f"✅ XGBoost carregado - {type(xgb_model).__name__}")
        
        with open(os.path.join(pasta_modelos, 'random_forest_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
            print(f"✅ Random Forest carregado - {type(rf_model).__name__}")
        
        with open(os.path.join(pasta_modelos, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
            print(f"✅ Scaler carregado - {type(scaler).__name__}")
        
        print(f"\n✅ TODOS OS MODELOS CARREGADOS COM SUCESSO!")
        
    except Exception as e:
        print(f"❌ Erro verificando modelos: {e}")
        return False
    
    # 4. Verificar banco de dados
    print(f"\n📊 VERIFICANDO BANCO DE DADOS:")
    db_path = 'royal_supreme_enhanced.db'
    
    if not os.path.exists(db_path):
        print(f"⚠️ Banco de dados {db_path} não encontrado")
    else:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Contar operações totais
            cursor.execute("SELECT COUNT(*) FROM operacoes")
            total_ops = cursor.fetchone()[0]
            
            # Contar operações com resultado
            cursor.execute("SELECT COUNT(*) FROM operacoes WHERE resultado IN ('WIN_M1', 'WIN_GALE', 'LOSS')")
            ops_com_resultado = cursor.fetchone()[0]
            
            # Últimas operações
            cursor.execute("""
                SELECT resultado, COUNT(*) 
                FROM operacoes 
                WHERE resultado IN ('WIN_M1', 'WIN_GALE', 'LOSS')
                GROUP BY resultado
            """)
            distribuicao = cursor.fetchall()
            
            conn.close()
            
            print(f"✅ Total de operações: {total_ops}")
            print(f"✅ Operações com resultado: {ops_com_resultado}")
            print(f"📈 Distribuição de resultados:")
            for resultado, count in distribuicao:
                print(f"   {resultado}: {count}")
                
        except Exception as e:
            print(f"⚠️ Erro acessando banco: {e}")
    
    # 5. Status final
    print(f"\n" + "="*50)
    print(f"🎯 STATUS FINAL DO TREINO DA IA:")
    print(f"="*50)
    
    if len(arquivos_encontrados) == len(arquivos_necessarios):
        print(f"✅ TREINO CONCLUÍDO COM SUCESSO!")
        print(f"🤖 Modelos XGBoost e Random Forest prontos")
        print(f"📊 Scaler e metadata salvos")
        print(f"🚀 IA Anti-Loss pronta para integração!")
        
        print(f"\n🎯 PRÓXIMOS PASSOS:")
        print(f"   1. Execute o sistema principal")
        print(f"   2. A IA será carregada automaticamente")
        print(f"   3. Sinais serão validados pela IA antes de operar")
        
        return True
    else:
        print(f"❌ TREINO INCOMPLETO!")
        print(f"💡 Execute novamente: python ai_model_trainer.py")
        return False

if __name__ == "__main__":
    verificar_treino_ia()