#!/usr/bin/env python3
"""
Script para ejecutar Llama3 directamente en el servidor remoto vía SSH
"""

import json
import os
import sys
import time
import paramiko
from typing import Dict, Any, Tuple, Optional

class RemoteSSHExecutor:
    """Clase para ejecutar comandos remotamente vía SSH"""
    
    def __init__(self, credentials_file: str = "ssh_credentials.json"):
        self.credentials_file = credentials_file
        self.credentials = None
        self.jump_client = None
        self.target_client = None
        self.channel = None
        
        self._load_credentials()
        
    def _load_credentials(self) -> bool:
        """Cargar credenciales desde archivo"""
        try:
            with open(self.credentials_file, 'r') as f:
                self.credentials = json.load(f)
            print("✅ Credenciales cargadas desde ssh_credentials.json")
            return True
        except Exception as e:
            print(f"❌ Error cargando credenciales: {e}")
            return False
    
    def connect(self) -> bool:
        """Establecer conexión SSH"""
        if not self.credentials:
            return False
            
        print("🔗 Conectando al servidor remoto...")
        
        try:
            ssh_config = self.credentials['ssh_config']
            jump_host = ssh_config['jump_host']
            target_host = ssh_config['target_host']
            
            print("✅ Credenciales verificadas correctamente")
            print(f"   Jump host: {jump_host['username']}@{jump_host['host']}:{jump_host['port']}")
            print(f"   Target host: {target_host['username']}@{target_host['host']}")
            print(f"   Modelo: {self.credentials['model_config']['model']}")
            
            # Conectar al jump host
            self.jump_client = paramiko.SSHClient()
            self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.jump_client.connect(
                hostname=jump_host['host'],
                port=jump_host['port'],
                username=jump_host['username'],
                password=jump_host['password']
            )
            
            # Configurar el túnel
            jump_transport = self.jump_client.get_transport()
            dest_addr = (target_host['host'], target_host['port'])
            local_addr = ('', 0)  # bind to any local port
            self.channel = jump_transport.open_channel(
                "direct-tcpip", dest_addr, local_addr
            )
            
            # Conectar al target host
            self.target_client = paramiko.SSHClient()
            self.target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.target_client.connect(
                hostname=target_host['host'],
                port=target_host['port'],
                username=target_host['username'],
                password=target_host['password'],
                sock=self.channel
            )
            
            print("✅ Conexión SSH establecida")
            return True
            
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            self.close()
            return False
    
    def run_command(self, command: str, timeout: int = 30) -> Tuple[bool, str]:
        """Ejecutar comando en el servidor remoto"""
        if not self.target_client:
            return False, "No hay conexión SSH"
            
        try:
            stdin, stdout, stderr = self.target_client.exec_command(command, timeout=timeout)
            
            # Para comandos de ollama, necesitamos manejar la salida de manera diferente
            if "ollama run" in command:
                # Dar tiempo para que ollama procese
                time.sleep(2)
                
            output = stdout.read().decode()
            error = stderr.read().decode()
            
            if error and "warning" not in error.lower():
                print(f"⚠️ Error: {error}")
                return False, error
                
            return True, output
            
        except Exception as e:
            print(f"❌ Error ejecutando comando: {e}")
            return False, str(e)
    
    def test_llama_interactive(self) -> bool:
        """Probar Llama3 usando un enfoque interactivo"""
        print("   🧪 Probando ejecución interactiva...")
        
        try:
            # Crear una sesión interactiva
            channel = self.target_client.invoke_shell()
            
            # Enviar comando
            channel.send("ollama run llama3.1\n")
            time.sleep(3)  # Esperar a que se inicie
            
            # Enviar prompt simple
            channel.send("Hi\n")
            time.sleep(5)  # Esperar respuesta
            
            # Leer respuesta
            output = ""
            while channel.recv_ready():
                output += channel.recv(1024).decode()
            
            # Salir
            channel.send("exit\n")
            channel.close()
            
            if output and len(output) > 10:
                print("✅ Llama3 está funcionando correctamente")
                print(f"   Respuesta de prueba:\n{output[-200:]}")
                return True
            else:
                print("❌ No se recibió respuesta válida")
                return False
                
        except Exception as e:
            print(f"❌ Error en prueba interactiva: {e}")
            return False
    
    def test_llama(self) -> bool:
        """Probar que Llama3 está funcionando"""
        print("\n🧪 Probando Llama3...")
        
        # Verificar que Ollama está instalado
        success, output = self.run_command("which ollama")
        if not success or "ollama" not in output:
            print("❌ Ollama no está instalado en el servidor")
            return False
            
        ollama_path = output.strip()
        print(f"✅ Ollama encontrado en: {ollama_path}")
        
        # Verificar estado del servicio
        print("   🔍 Verificando servicio Ollama...")
        success, output = self.run_command("ps aux | grep ollama | grep -v grep")
        
        if not success or not output:
            print("⚠️ Servicio Ollama no está activo")
            print("   🔄 Iniciando servicio...")
            success, _ = self.run_command(f"{ollama_path} serve &")
            if not success:
                print("❌ Error iniciando servicio Ollama")
                return False
                
            # Esperar a que el servicio esté listo
            print("   ⏳ Esperando a que el servicio esté listo...")
            time.sleep(30)  # Esperar 30 segundos
        else:
            print("✅ Servicio Ollama está activo")
            
        # Verificar que Llama3 está disponible
        success, output = self.run_command(f"{ollama_path} list")
        if not success:
            print("❌ Error verificando modelos disponibles")
            return False
            
        if "llama3.1" not in output.lower():
            print("❌ Modelo llama3.1 no está disponible")
            print("   🔄 Instalando llama3.1 (esto puede tomar varios minutos)...")
            success, _ = self.run_command(f"{ollama_path} pull llama3.1", timeout=600)
            if not success:
                print("❌ Error instalando llama3.1")
                return False
            
            # Verificar instalación
            success, output = self.run_command(f"{ollama_path} list")
            if not success or "llama3.1" not in output.lower():
                print("❌ Error verificando instalación de llama3.1")
                return False
        
        # Probar Llama3 con método interactivo
        return self.test_llama_interactive()
    
    def run_ollama_structured_command(self, payload: Dict[str, Any], timeout: int = 30) -> Tuple[bool, str]:
        """Ejecutar un comando estructurado con ollama usando el formato JSON especificado"""
        try:
            # Extraer información del payload
            modelo = payload.get("model", "llama3.1")
            messages = payload.get("messages", [])
            temperatura = payload.get("temperature", 0.7)
            
            # Construir el prompt completo a partir de los mensajes
            prompt_parts = []
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            # Combinar todas las partes del prompt
            full_prompt = "\n\n".join(prompt_parts)
            
            # Crear una sesión interactiva
            channel = self.target_client.invoke_shell()
            
            # Enviar comando para iniciar ollama con el modelo específico
            channel.send(f"ollama run {modelo}\n")
            time.sleep(3)  # Esperar a que se inicie
            
            # Limpiar el buffer
            while channel.recv_ready():
                channel.recv(1024)
            
            # Enviar el prompt estructurado
            channel.send(f"{full_prompt}\n")
            time.sleep(10)  # Esperar más tiempo para la respuesta estructurada
            
            # Leer toda la respuesta
            output = ""
            max_attempts = 15
            attempts = 0
            
            while attempts < max_attempts:
                if channel.recv_ready():
                    data = channel.recv(1024).decode()
                    output += data
                    attempts = 0  # Reset counter if we're still receiving data
                else:
                    time.sleep(0.5)
                    attempts += 1
            
            # Salir de ollama
            channel.send("/bye\n")
            time.sleep(1)
            channel.close()
            
            # Extraer solo la respuesta generada
            if output:
                # Buscar la respuesta después del prompt
                lines = output.split('\n')
                response_started = False
                response_lines = []
                
                for line in lines:
                    # Si encontramos partes del prompt, comenzamos a buscar la respuesta
                    if not response_started and ("User:" in line or "Assistant:" in line):
                        response_started = True
                        continue
                    
                    # Si ya comenzamos a recolectar y encontramos el prompt de ollama, paramos
                    if response_started and (">>>" in line or "Send a message" in line):
                        break
                    
                    # Recolectar líneas de respuesta
                    if response_started and line.strip():
                        # Filtrar líneas que no son parte de la respuesta
                        if not any(marker in line for marker in [">>>", "Send a message", "Use /bye"]):
                            response_lines.append(line.strip())
                
                response = '\n'.join(response_lines).strip()
                
                # Limpiar respuesta de artefactos comunes
                response = response.replace(">>> ", "").replace("Send a message (/? for help)", "")
                
                if response and len(response) > 10:  # Asegurar que hay contenido real
                    return True, response
                else:
                    return False, "Respuesta vacía o muy corta"
            else:
                return False, "No se recibió salida"
                
        except Exception as e:
            print(f"❌ Error ejecutando comando ollama estructurado: {e}")
            return False, str(e)
    
    def run_ollama_command(self, prompt: str, timeout: int = 30) -> Tuple[bool, str]:
        """Ejecutar un comando específico con ollama"""
        try:
            # Crear una sesión interactiva
            channel = self.target_client.invoke_shell()
            
            # Enviar comando para iniciar ollama
            channel.send("ollama run llama3.1\n")
            time.sleep(3)  # Esperar a que se inicie
            
            # Limpiar el buffer
            while channel.recv_ready():
                channel.recv(1024)
            
            # Enviar el prompt
            channel.send(f"{prompt}\n")
            time.sleep(8)  # Esperar más tiempo para la respuesta
            
            # Leer toda la respuesta
            output = ""
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                if channel.recv_ready():
                    data = channel.recv(1024).decode()
                    output += data
                    attempts = 0  # Reset counter if we're still receiving data
                else:
                    time.sleep(0.5)
                    attempts += 1
            
            # Salir de ollama
            channel.send("/bye\n")
            time.sleep(1)
            channel.close()
            
            # Extraer solo la respuesta generada (después del prompt)
            if output:
                # Buscar el patrón del prompt en la salida
                lines = output.split('\n')
                response_started = False
                response_lines = []
                
                for line in lines:
                    # Si encontramos el prompt, comenzamos a recolectar la respuesta
                    if not response_started and prompt.strip() in line:
                        response_started = True
                        continue
                    
                    # Si ya comenzamos a recolectar y encontramos el prompt de ollama, paramos
                    if response_started and (">>>" in line or "Send a message" in line):
                        break
                    
                    # Recolectar líneas de respuesta
                    if response_started and line.strip():
                        response_lines.append(line.strip())
                
                response = '\n'.join(response_lines).strip()
                
                if response and len(response) > 5:  # Asegurar que hay contenido real
                    return True, response
                else:
                    return False, "Respuesta vacía o muy corta"
            else:
                return False, "No se recibió salida"
                
        except Exception as e:
            print(f"❌ Error ejecutando comando ollama: {e}")
            return False, str(e)
    
    def close(self):
        """Cerrar conexiones SSH"""
        print("🧹 Cerrando conexión SSH...")
        
        if self.target_client:
            self.target_client.close()
            
        if self.channel:
            self.channel.close()
            
        if self.jump_client:
            self.jump_client.close()
            
        print("✅ Conexión SSH cerrada")

def main():
    """Función principal"""
    print("🚀 Ejecutor Remoto SSH para Llama3")
    print("=" * 50)
    
    executor = RemoteSSHExecutor()
    
    try:
        if executor.connect():
            if executor.test_llama():
                print("\n🎉 ¡Conexión y Llama3 funcionando correctamente!")
                print("   Puedes usar esta conexión para ejecutar comandos remotos.")
                
                # Ejemplo de uso
                print("\n📝 Ejemplo de uso:")
                success, output = executor.run_command('echo "¿Qué es COVID-19?" | ollama run llama2')
                if success:
                    print(f"Respuesta:\n{output}")
            else:
                print("\n❌ Llama3 no está funcionando correctamente")
        else:
            print("\n❌ No se pudo establecer la conexión SSH")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario")
    finally:
        executor.close()

if __name__ == "__main__":
    main() 