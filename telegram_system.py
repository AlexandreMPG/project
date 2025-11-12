#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
👑 TELEGRAM SYSTEM - ROYAL SUPREME ENHANCED 👑
💎 SESSION MANAGER + ADMIN COMMANDS + ROYAL TELEGRAM
🔥 SISTEMA TELEGRAM COMPLETO + ELITE TRADER ENHANCED
"""

import time
import datetime
import threading
import requests
from typing import Dict, Any, List
from collections import defaultdict
from config_royal import ConfigRoyalSupremeEnhanced, ROYAL_TELEGRAM, TIMEZONE

class RoyalSessionManagerSupremeEnhanced:
    
    def __init__(self):
        self.sessao_ativa = False
        self.sinais_sessao_atual = 0
        self.timestamp_inicio_sessao = 0
        self.timestamp_proximo_aviso = 0
        self.timestamp_proxima_sessao = int(time.time()) + 60
        self.aviso_enviado = False
        
    def pode_emitir_sinal_free(self) -> bool:
        """Verifica se pode emitir sinal FREE"""
        tempo_atual = int(time.time())
        
        # Iniciar nova sessão se necessário
        if not self.sessao_ativa and tempo_atual >= self.timestamp_proxima_sessao:
            self._iniciar_nova_sessao()
        
        if self.sessao_ativa:
            tempo_sessao = tempo_atual - self.timestamp_inicio_sessao
            if tempo_sessao > (ConfigRoyalSupremeEnhanced.SESSAO_FREE_DURACAO * 60):
                self._finalizar_sessao()
                return False
            
            return self.sinais_sessao_atual < ConfigRoyalSupremeEnhanced.SESSAO_FREE_SINAIS
        
        return False
    
    def _iniciar_nova_sessao(self):
        """Inicia uma nova sessão FREE"""
        tempo_atual = int(time.time())
        self.sessao_ativa = True
        self.sinais_sessao_atual = 0
        self.timestamp_inicio_sessao = tempo_atual
        self.aviso_enviado = False
        
        print(f"👑 NOVA SESSÃO ROYAL ENHANCED INICIADA!")
        print(f"🏆 {ConfigRoyalSupremeEnhanced.SESSAO_FREE_SINAIS} sinais FREE por {ConfigRoyalSupremeEnhanced.SESSAO_FREE_DURACAO} minutos")
    
    def _finalizar_sessao(self):
        """Finaliza a sessão atual"""
        self.sessao_ativa = False
        tempo_atual = int(time.time())
        
        self.timestamp_proxima_sessao = tempo_atual + (ConfigRoyalSupremeEnhanced.SESSAO_FREE_INTERVALO * 60)
        self.timestamp_proximo_aviso = self.timestamp_proxima_sessao - (ConfigRoyalSupremeEnhanced.AVISO_SESSAO_MINUTOS * 60)
        
        intervalo_min = ConfigRoyalSupremeEnhanced.SESSAO_FREE_INTERVALO
        print(f"🛑 SESSÃO ROYAL ENHANCED FINALIZADA - Próxima em {intervalo_min} minutos")
    
    def registrar_sinal_emitido(self):
        """Registra que um sinal foi emitido"""
        if self.sessao_ativa:
            self.sinais_sessao_atual += 1
            print(f"👑 FREE: {self.sinais_sessao_atual}/{ConfigRoyalSupremeEnhanced.SESSAO_FREE_SINAIS} enviados")
    
    def verificar_aviso_sessao(self) -> bool:
        """Verifica se deve enviar aviso de nova sessão"""
        tempo_atual = int(time.time())
        
        if (not self.aviso_enviado and 
            tempo_atual >= self.timestamp_proximo_aviso and 
            tempo_atual < self.timestamp_proxima_sessao):
            self.aviso_enviado = True
            return True
        
        return False
    
    def get_status(self) -> Dict:
        """Retorna status da sessão"""
        tempo_atual = int(time.time())
        
        if self.sessao_ativa:
            tempo_restante = max(0, (ConfigRoyalSupremeEnhanced.SESSAO_FREE_DURACAO * 60) - (tempo_atual - self.timestamp_inicio_sessao))
            return {
                'ativa': True,
                'sinais_enviados': self.sinais_sessao_atual,
                'sinais_restantes': ConfigRoyalSupremeEnhanced.SESSAO_FREE_SINAIS - self.sinais_sessao_atual,
                'tempo_restante_min': tempo_restante // 60
            }
        else:
            tempo_para_proxima = max(0, self.timestamp_proxima_sessao - tempo_atual)
            return {
                'ativa': False,
                'tempo_para_proxima_min': tempo_para_proxima // 60
            }

class TelegramAdminCommandsEnhanced:
    
    def __init__(self, relatorios_system):
        self.relatorios = relatorios_system
        self.bot_token = ROYAL_TELEGRAM['free_token']
        self.admin_ids = ROYAL_TELEGRAM.get('admin_ids', [])
        self.last_update_id = 0
        self.running = False
        
        if self.bot_token and self.admin_ids:
            print("🤖 Sistema de Comandos Admin Telegram ENHANCED ATIVO")
    
    def start(self):
        """Inicia o sistema de comandos admin"""
        if self.bot_token and self.admin_ids and not self.running:
            self.running = True
            thread = threading.Thread(target=self._check_messages_loop, daemon=True)
            thread.start()
    
    def _check_messages_loop(self):
        """Loop principal de verificação de mensagens"""
        while self.running:
            try:
                self._check_new_messages()
                time.sleep(2)
            except Exception as e:
                time.sleep(5)
    
    def _check_new_messages(self):
        """Verifica novas mensagens do Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {'offset': self.last_update_id + 1, 'timeout': 1}
            
            response = requests.get(url, params=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('ok') and data.get('result'):
                    for update in data['result']:
                        self._process_update(update)
                        self.last_update_id = update['update_id']
        except:
            pass
    
    def _process_update(self, update):
        """Processa uma atualização do Telegram"""
        try:
            if 'message' not in update:
                return
            
            message = update['message']
            user_id = message.get('from', {}).get('id')
            chat_id = message.get('chat', {}).get('id')
            text = message.get('text', '').strip()
            
            # Só aceitar comandos em chat privado
            if chat_id != user_id:
                return
            
            # Só aceitar comandos que começam com /
            if not text.startswith('/'):
                return
            
            # Verificar se é admin
            if user_id not in self.admin_ids:
                self._send_message(chat_id, "❌ Acesso negado. Você não é um admin autorizado.")
                return
            
            self._handle_command(chat_id, text.lower())
                
        except Exception as e:
            pass
    
    def _handle_command(self, chat_id: int, command: str):
        """Processa um comando admin"""
        try:
            if command == '/start' or command == '/help':
                self._send_help(chat_id)
            
            elif command == '/relatorio1' or command == '/r1':
                relatorio = self.relatorios.gerar_relatorio_completo()
                self._send_long_message(chat_id, relatorio)
            
            elif command == '/resumo1' or command == '/rs1':
                resumo = self.relatorios.gerar_relatorio_resumido()
                self._send_message(chat_id, f"```\n{resumo}\n```")
            
            elif command == '/stats1' or command == '/s1':
                stats = self._gerar_stats_rapidas()
                self._send_message(chat_id, f"```\n{stats}\n```")
            
            elif command == '/status1':
                status = self._gerar_status_sistema()
                self._send_message(chat_id, status)
            
            elif command == '/db1':
                if hasattr(self.relatorios, 'db_manager'):
                    db_stats = self.relatorios.db_manager.get_relatorio_inteligencia()
                    self._send_message(chat_id, f"```\n{db_stats}\n```")
                else:
                    self._send_message(chat_id, "❌ Database manager não disponível")
            
            else:
                self._send_message(chat_id, "❌ Comando não reconhecido. Use /help para ver comandos disponíveis.")
                
        except Exception as e:
            self._send_message(chat_id, f"❌ Erro executando comando: {str(e)}")
    
    def _send_long_message(self, chat_id: int, text: str):
        """Envia mensagem longa dividida em partes"""
        try:
            max_length = 4000
            
            if len(text) <= max_length:
                self._send_message(chat_id, f"```\n{text}\n```")
            else:
                parts = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                for i, part in enumerate(parts, 1):
                    header = f"📊 **RELATÓRIO PARTE {i}/{len(parts)}**\n\n"
                    self._send_message(chat_id, f"{header}```\n{part}\n```")
                    time.sleep(1)
        except Exception as e:
            self._send_message(chat_id, f"❌ Erro enviando relatório: {str(e)}")
    
    def _send_help(self, chat_id: int):
        """Envia ajuda dos comandos"""
        help_text = """🤖 **COMANDOS ADMIN CIPHER ENHANCED + AI**

📊 **RELATÓRIOS:**
/relatorio1 ou /r1 - Relatório completo
/resumo1 ou /rs1 - Resumo rápido  
/stats1 ou /s1 - Estatísticas atuais
/db1 - Relatório database AI

🔧 **SISTEMA:**
/status1 - Status do sistema
/help - Esta ajuda

👑 **Sistema Royal Supreme Enhanced + Database AI**"""
        
        self._send_message(chat_id, help_text)
    
    def _gerar_stats_rapidas(self) -> str:
        """Gera estatísticas rápidas"""
        stats = self.relatorios.stats_globais
        
        total_trades = stats['wins_m1'] + stats['wins_gale'] + stats['losses']
        win_rate = ((stats['wins_m1'] + stats['wins_gale']) / total_trades * 100) if total_trades > 0 else 0
        
        return f"""👑 STATS RÁPIDAS ROYAL SUPREME ENHANCED + AI

🎯 Total Sinais: {stats['total_sinais']}
📊 Win Rate: {win_rate:.1f}%
🏆 Wins M1: {stats['wins_m1']}
🟡 Wins Gale: {stats['wins_gale']}
❌ Losses: {stats['losses']}
⚡ AUTO CALIBRADOR: {stats['auto_calibrador_usado']}
💎 Enhanced Weight: {stats['enhanced_weight_aplicado']}
🎯 Trend Opportunities: {stats.get('trend_opportunities', 0)}
🔄 Pullback Opportunities: {stats.get('pullback_opportunities', 0)}
🌊 Elliott Opportunities: {stats.get('elliott_opportunities', 0)}
⚡ Elite Opportunities: {stats.get('elite_opportunities', 0)}
🌊 Wave Opportunities: {stats.get('wave_opportunities', 0)}

👑 Elite Business Enhanced + AI!"""
    
    def _gerar_status_sistema(self) -> str:
        """Gera status do sistema"""
        tempo_operacao = int(time.time()) - self.relatorios.stats_globais['tempo_inicio']
        horas = tempo_operacao // 3600
        minutos = (tempo_operacao % 3600) // 60
        
        return f"""🔧 **STATUS SISTEMA ENHANCED + AI**

⏰ **Tempo Online:** {horas}h {minutos}m
🎯 **Total Sinais:** {self.relatorios.stats_globais['total_sinais']}
🤖 **Bot Status:** ✅ Online
📱 **Telegram:** ✅ Funcionando
🎯 **Enhanced Features:** ✅ Ativo
🗄️ **Database AI:** ✅ Funcionando

👑 **Sistema Operacional Enhanced + AI**"""
    
    def _send_message(self, chat_id: int, text: str):
        """Envia mensagem para o Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=5)
            
        except Exception as e:
            pass

class RoyalTelegramSupremeEnhanced:
    
    def __init__(self):
        self.free_enabled = ROYAL_TELEGRAM['enabled']
        self.vip_enabled = ROYAL_TELEGRAM['enabled']
        self.sinal_numero = 0
        self.royal_stats = {'total': 0, 'wins': 0, 'losses': 0}
        self.wins_consecutivos = 0
        
        if ROYAL_TELEGRAM['free_token'] and ROYAL_TELEGRAM['free_chat']:
            self.free_enabled = True
            print("👑 Telegram FREE auto-configurado - Royal Supreme Enhanced + AI")
        
        if ROYAL_TELEGRAM['vip_token'] and ROYAL_TELEGRAM['vip_chat']:
            self.vip_enabled = True
            print("💎 Telegram VIP auto-configurado - Royal Supreme Enhanced + AI")
    
    def enviar_sinal(self, sinal_data: Dict, is_free: bool = True):
        """Envia sinal para o Telegram"""
        self.sinal_numero += 1
        self.royal_stats['total'] += 1
        
        par_nome = sinal_data['par']
        tipo = sinal_data['tipo']
        score = sinal_data['score']
        enhanced_features = sinal_data.get('enhanced_features', [])
        
        agora_br = datetime.datetime.now(TIMEZONE)
        proxima_vela = agora_br.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        horario_proxima_vela = proxima_vela.strftime('%H:%M')
        
        par_upper = par_nome.upper()
        
        if 'CALL' in tipo:
            emoji_tipo = '🟢'
            tipo_texto = 'CALL'
        else:
            emoji_tipo = '🔴'
            tipo_texto = 'PUT'
        
        enhanced_text = ""
        if enhanced_features:
            enhanced_text = f"\n🎯 {' • '.join(enhanced_features[:2])}"
        
        # Adicionar indicador de AI se for relevante
        ai_indicator = "\n🤖 AI Enhanced" if enhanced_features else ""
        
        mensagem = f"""👑 ELITE #{self.sinal_numero}

**{par_upper}**
{emoji_tipo} {tipo_texto}
{horario_proxima_vela} • {score:.0f}%{enhanced_text}{ai_indicator}

Elite Trader Enhanced + AI"""
        
        if is_free and self.free_enabled:
            self._enviar_telegram(ROYAL_TELEGRAM['free_token'], ROYAL_TELEGRAM['free_chat'], mensagem)
        
        if self.vip_enabled:
            self._enviar_telegram(ROYAL_TELEGRAM['vip_token'], ROYAL_TELEGRAM['vip_chat'], mensagem)
    
    def enviar_resultado(self, par: str, tipo: str, resultado: str):
        """Envia resultado da operação"""
        par_upper = par.upper()
        
        if resultado in ['WIN_M1', 'WIN_GALE']:
            self.royal_stats['wins'] += 1
            self.wins_consecutivos += 1
            
            if resultado == 'WIN_M1':
                mensagem = f"""🏆 RESULTADO #{self.sinal_numero}

**{par_upper}**
✅ WIN M1

Elite Result Enhanced + AI"""
            else:
                mensagem = f"""🏆 RESULTADO #{self.sinal_numero}

**{par_upper}**
🟡 WIN GALE

Elite Result Enhanced + AI"""
                
        else:
            self.royal_stats['losses'] += 1
            self.wins_consecutivos = 0
            mensagem = f"""🏆 RESULTADO #{self.sinal_numero}

**{par_upper}**
❌ LOSS

Elite Result Enhanced + AI"""
        
        if self.free_enabled:
            self._enviar_telegram(ROYAL_TELEGRAM['free_token'], ROYAL_TELEGRAM['free_chat'], mensagem)
        if self.vip_enabled:
            self._enviar_telegram(ROYAL_TELEGRAM['vip_token'], ROYAL_TELEGRAM['vip_chat'], mensagem)
    
    def enviar_aviso_sessao(self):
        """Envia aviso de nova sessão"""
        proximo_horario = (datetime.datetime.now(TIMEZONE) + datetime.timedelta(minutes=ConfigRoyalSupremeEnhanced.AVISO_SESSAO_MINUTOS)).strftime('%H:%M')
        
        mensagem = f"""🚨 AVISO SESSÃO

👑 ROYAL FREE ENHANCED + AI

NOVA SESSÃO EM {ConfigRoyalSupremeEnhanced.AVISO_SESSAO_MINUTOS} MINUTOS
{ConfigRoyalSupremeEnhanced.SESSAO_FREE_SINAIS} sinais loading...

Elite Session Enhanced + AI"""
        
        if self.free_enabled:
            self._enviar_telegram(ROYAL_TELEGRAM['free_token'], ROYAL_TELEGRAM['free_chat'], mensagem)
    
    def _enviar_telegram(self, token: str, chat_id: str, mensagem: str):
        """Envia mensagem para o Telegram"""
        try:
            if not token or not chat_id:
                return
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': mensagem,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=5)
                
        except Exception as e:
            pass

print("✅ TELEGRAM SYSTEM - ROYAL SUPREME ENHANCED + AI CARREGADO!")
