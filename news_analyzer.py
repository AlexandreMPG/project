#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📰 NEWS ANALYZER - SISTEMA DE NOTÍCIAS RSS FREE
🔥 SALVAR COMO: news_analyzer.py
💎 INTEGRAÇÃO COM CIPHER ROYAL SUPREME ENHANCED
"""

import requests
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import re

class NewsImpactAnalyzer:
    """📰 ANALISADOR DE IMPACTO DE NOTÍCIAS FREE"""
    
    def __init__(self):
        self.noticias_cache = deque(maxlen=50)
        self.ultima_atualizacao = 0
        self.impacto_global = "NORMAL"
        self.sentiment_score = 0  # -100 (muito bearish) a +100 (muito bullish)
        
        # ⚙️ CONFIGURAÇÕES
        self.ATIVO = True  # Mude para False para desativar
        self.INTERVALO_MINUTOS = 10  # Atualizar a cada 10 minutos
        
        # 🌐 FEEDS RSS GRATUITOS
        self.RSS_FEEDS = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cryptonews.com/news/feed/",
            "https://www.crypto-news.net/feed/",
            "https://decrypt.co/feed"
        ]
        
        # 🎯 PALAVRAS-CHAVE DE IMPACTO
        self.KEYWORDS_BULLISH = [
            'adoption', 'bull', 'surge', 'rally', 'pump', 'breakthrough', 'partnership', 
            'investment', 'approval', 'upgrade', 'launch', 'institutional', 'etf',
            'buy', 'rise', 'gains', 'positive', 'optimistic', 'moon'
        ]
        
        self.KEYWORDS_BEARISH = [
            'crash', 'dump', 'bear', 'regulation', 'ban', 'hack', 'scam', 'decline', 
            'drop', 'fall', 'correction', 'sell-off', 'liquidation', 'fear', 'panic',
            'negative', 'concern', 'warning', 'risk'
        ]
        
        self.KEYWORDS_CRITICAL = [
            'emergency', 'halt', 'suspend', 'investigation', 'fraud', 'bankruptcy',
            'sec', 'cftc', 'lawsuit', 'fine', 'penalty', 'arrest'
        ]
        
        # 🪙 MOEDAS PARA MONITORAR
        self.COINS_KEYWORDS = {
            'btcusdt': ['bitcoin', 'btc'],
            'ethusdt': ['ethereum', 'eth', 'ether'],
            'solusdt': ['solana', 'sol'],
            'adausdt': ['cardano', 'ada'],
            'xrpusdt': ['ripple', 'xrp']
        }
    
    def buscar_noticias(self) -> List[Dict]:
        """🔍 Busca notícias dos feeds RSS"""
        if not self.ATIVO:
            return []
        
        # Verificar se precisa atualizar
        now = time.time()
        if now - self.ultima_atualizacao < (self.INTERVALO_MINUTOS * 60):
            return list(self.noticias_cache)
        
        todas_noticias = []
        
        for feed_url in self.RSS_FEEDS:
            try:
                # Timeout curto para não travar o sistema
                response = requests.get(feed_url, timeout=5, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    
                    # Buscar items do RSS
                    items = root.findall('.//item')[:5]  # Apenas 5 mais recentes por feed
                    
                    for item in items:
                        titulo = item.find('title')
                        descricao = item.find('description')
                        link = item.find('link')
                        pub_date = item.find('pubDate')
                        
                        if titulo is not None:
                            noticia = {
                                'titulo': titulo.text if titulo.text else '',
                                'descricao': descricao.text if descricao else '',
                                'link': link.text if link else '',
                                'data': pub_date.text if pub_date else '',
                                'fonte': feed_url.split('//')[1].split('/')[0],
                                'timestamp': now
                            }
                            
                            # Analisar impacto da notícia
                            noticia.update(self._analisar_impacto_noticia(noticia))
                            todas_noticias.append(noticia)
                
            except Exception as e:
                # Falha silenciosa para não atrapalhar o sistema principal
                continue
        
        # Atualizar cache
        self.noticias_cache.clear()
        self.noticias_cache.extend(todas_noticias[-20:])  # Manter apenas 20 mais recentes
        self.ultima_atualizacao = now
        
        # Calcular impacto global
        self._calcular_impacto_global()
        
        return todas_noticias
    
    def _analisar_impacto_noticia(self, noticia: Dict) -> Dict:
        """🎯 Analisa o impacto de uma notícia específica"""
        texto_completo = f"{noticia['titulo']} {noticia['descricao']}".lower()
        
        # Contar palavras-chave
        bullish_count = sum(1 for word in self.KEYWORDS_BULLISH if word in texto_completo)
        bearish_count = sum(1 for word in self.KEYWORDS_BEARISH if word in texto_completo)
        critical_count = sum(1 for word in self.KEYWORDS_CRITICAL if word in texto_completo)
        
        # Determinar impacto
        if critical_count > 0:
            impacto = "CRITICAL"
            sentimento = "VERY_BEARISH"
            score = -80
        elif bearish_count > bullish_count + 1:
            impacto = "HIGH" if bearish_count >= 3 else "MEDIUM"
            sentimento = "BEARISH"
            score = -40 if impacto == "HIGH" else -20
        elif bullish_count > bearish_count + 1:
            impacto = "HIGH" if bullish_count >= 3 else "MEDIUM"
            sentimento = "BULLISH"
            score = 40 if impacto == "HIGH" else 20
        else:
            impacto = "LOW"
            sentimento = "NEUTRAL"
            score = 0
        
        # Identificar moedas afetadas
        moedas_afetadas = []
        for par, keywords in self.COINS_KEYWORDS.items():
            if any(keyword in texto_completo for keyword in keywords):
                moedas_afetadas.append(par)
        
        return {
            'impacto': impacto,
            'sentimento': sentimento,
            'score': score,
            'moedas_afetadas': moedas_afetadas,
            'bullish_words': bullish_count,
            'bearish_words': bearish_count,
            'critical_words': critical_count
        }
    
    def _calcular_impacto_global(self):
        """📊 Calcula impacto global das notícias recentes"""
        if not self.noticias_cache:
            self.impacto_global = "NORMAL"
            self.sentiment_score = 0
            return
        
        # Calcular score médio das últimas 10 notícias
        scores = [n.get('score', 0) for n in list(self.noticias_cache)[-10:]]
        media_score = sum(scores) / len(scores) if scores else 0
        
        self.sentiment_score = media_score
        
        # Determinar impacto global
        critical_news = sum(1 for n in self.noticias_cache if n.get('impacto') == 'CRITICAL')
        high_impact = sum(1 for n in self.noticias_cache if n.get('impacto') == 'HIGH')
        
        if critical_news >= 2:
            self.impacto_global = "CRITICAL"
        elif critical_news >= 1 or high_impact >= 3:
            self.impacto_global = "HIGH"
        elif high_impact >= 1 or abs(media_score) > 15:
            self.impacto_global = "MEDIUM"
        else:
            self.impacto_global = "NORMAL"
    
    def get_impacto_para_par(self, par: str) -> Dict:
        """🎯 Retorna impacto específico para um par"""
        if not self.ATIVO:
            return {'impacto': 'NORMAL', 'score_ajuste': 0, 'motivo': 'News desativado'}
        
        # Buscar notícias recentes (se necessário)
        self.buscar_noticias()
        
        # Analisar notícias que afetam este par especificamente
        noticias_par = [n for n in self.noticias_cache if par in n.get('moedas_afetadas', [])]
        
        if not noticias_par:
            # Usar impacto global se não há notícias específicas
            return {
                'impacto': self.impacto_global,
                'score_ajuste': self._calcular_ajuste_score(self.sentiment_score),
                'motivo': f'Impacto global: {self.impacto_global}',
                'sentiment_score': self.sentiment_score
            }
        
        # Calcular impacto específico do par
        score_par = sum(n.get('score', 0) for n in noticias_par[-3:])  # Últimas 3 notícias
        impacto_par = max([n.get('impacto', 'LOW') for n in noticias_par[-3:]], 
                         key=lambda x: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(x))
        
        return {
            'impacto': impacto_par,
            'score_ajuste': self._calcular_ajuste_score(score_par),
            'motivo': f'Notícias específicas {par}: {len(noticias_par)} encontradas',
            'sentiment_score': score_par,
            'noticias_recentes': len(noticias_par)
        }
    
    def _calcular_ajuste_score(self, sentiment_score: float) -> float:
        """📈 Converte sentiment score em ajuste para o score do sinal"""
        if sentiment_score > 50:
            return 30  # Muito bullish: +30 pontos
        elif sentiment_score > 20:
            return 15  # Bullish: +15 pontos
        elif sentiment_score < -50:
            return -30  # Muito bearish: -30 pontos
        elif sentiment_score < -20:
            return -15  # Bearish: -15 pontos
        else:
            return 0   # Neutro: sem ajuste
    
    def get_status_resumido(self) -> str:
        """📊 Retorna status resumido para logs"""
        if not self.ATIVO:
            return "📰 News: DESATIVADO"
        
        noticias_count = len(self.noticias_cache)
        return f"📰 News: {self.impacto_global} | Score: {self.sentiment_score:.0f} | {noticias_count} notícias"

print("✅ NEWS ANALYZER CARREGADO - SISTEMA DE NOTÍCIAS RSS FREE!")