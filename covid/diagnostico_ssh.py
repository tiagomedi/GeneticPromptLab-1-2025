#!/usr/bin/env python3
"""
Script de diagnóstico para conexión SSH y Llama3
"""

import json
import os
import sys
import paramiko
import time
from typing import Dict, Any, Tuple

class SSHDiagnostic:
    def __init__(self):
        self.creds_file = "ssh_credentials.json"
        self.credentials = None
        
    def check_credentials_file(self) -> bool:
        """Verificar archivo de credenciales"""
        print("\n============================================================")
        print("🔍 PASO 1: Verificando archivo de credenciales...")
        
        if not os.path.exists(self.creds_file):
            print(f"❌ No se encuentra el archivo {self.creds_file}")
            return False
            
        try:
            with open(self.creds_file, 'r') as f:
                self.credentials = json.load(f)
            print("✅ Archivo ssh_credentials.json encontrado y leído")
            
            print("📋 Estructura del archivo:")
            for key, value in self.credentials.items():
                print(f"   - {key}: {type(value)}")
            
            return True
            
        except json.JSONDecodeError:
            print("❌ Error decodificando JSON")
            return False
        except Exception as e:
            print(f"❌ Error leyendo archivo: {e}")
            return False
    
    def check_credentials_structure(self) -> bool:
        """Verificar estructura de credenciales"""
        print("\n============================================================")
        print("\n🔍 PASO 2: Verificando estructura de credenciales...")
        
        required_sections = ['ssh_config', 'model_config']
        missing_sections = []
        
        for section in required_sections:
            if section in self.credentials:
                print(f"✅ Sección encontrada: {section}")
            else:
                print(f"❌ Sección faltante: {section}")
                missing_sections.append(section)
        
        if missing_sections:
            print(f"\n❌ FALLO EN: Verificar estructura de credenciales")
            print("🛑 Diagnóstico detenido debido a fallo crítico")
            return False
            
        return True
    
    def check_ssh_connection(self) -> bool:
        """Verificar conexión SSH"""
        print("\n============================================================")
        print("\n🔍 PASO 3: Probando conexión SSH...")
        
        ssh_config = self.credentials['ssh_config']
        jump_host = ssh_config['jump_host']
        target_host = ssh_config['target_host']
        
        # Probar conexión al jump host
        print("\n📡 Conectando al jump host...")
        try:
            jump_client = paramiko.SSHClient()
            jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            jump_client.connect(
                hostname=jump_host['host'],
                port=jump_host['port'],
                username=jump_host['username'],
                password=jump_host['password']
            )
            print("✅ Conexión al jump host exitosa")
            
            # Probar conexión al target host
            print("\n📡 Conectando al target host...")
            target_client = paramiko.SSHClient()
            target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Configurar el túnel
            jump_transport = jump_client.get_transport()
            dest_addr = (target_host['host'], target_host['port'])
            local_addr = ('', 0)  # bind to any local port
            channel = jump_transport.open_channel(
                "direct-tcpip", dest_addr, local_addr
            )
            
            # Conectar al target host a través del túnel
            target_client.connect(
                hostname=target_host['host'],
                port=target_host['port'],
                username=target_host['username'],
                password=target_host['password'],
                sock=channel
            )
            print("✅ Conexión al target host exitosa")
            
            # Probar Llama3
            print("\n🤖 Verificando Llama3...")
            stdin, stdout, stderr = target_client.exec_command("ollama list")
            time.sleep(2)
            output = stdout.read().decode()
            
            if "llama2:13b" not in output.lower():
                print("❌ Modelo llama2:13b no está instalado")
                print("   🔄 Instalando llama2:13b (esto puede tomar varios minutos)...")
                
                # Intentar instalar
                cmd = 'ollama pull llama2:13b'
                stdin, stdout, stderr = target_client.exec_command(cmd)
                
                # Esperar hasta 10 minutos por la instalación
                for _ in range(60):  # 10 minutos en intervalos de 10 segundos
                    time.sleep(10)
                    if stdout.channel.exit_status_ready():
                        break
                
                # Verificar si se instaló
                stdin, stdout, stderr = target_client.exec_command("ollama list")
                time.sleep(2)
                output = stdout.read().decode()
                
                if "llama2:13b" not in output.lower():
                    print("❌ Error instalando llama2:13b")
                    return False
                else:
                    print("✅ llama2:13b instalado correctamente")
            else:
                print("✅ llama2:13b ya está instalado")
            
            # Probar ejecución
            print("\n🧪 Probando ejecución de Llama3...")
            cmd = 'echo "Hello" | ollama run llama2:13b'
            stdin, stdout, stderr = target_client.exec_command(cmd)
            time.sleep(5)
            response = stdout.read().decode()
            
            if response.strip():
                print("✅ Llama3 responde correctamente")
                print(f"📝 Respuesta de prueba:\n{response[:200]}...")
            else:
                print("❌ Llama3 no respondió")
            return True
            
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False
    
    def run_diagnostic(self):
        """Ejecutar diagnóstico completo"""
        print("🩺 DIAGNÓSTICO SSH COMPLETO")
        print("=" * 50)
        
        results = []
        
        # Paso 1: Verificar archivo
        success = self.check_credentials_file()
        results.append(("Verificar archivo de credenciales", success))
        if not success:
            self.show_summary(results)
            return
        
        # Paso 2: Verificar estructura
        success = self.check_credentials_structure()
        results.append(("Verificar estructura de credenciales", success))
        if not success:
            self.show_summary(results)
            return
        
        # Paso 3: Verificar conexión
        success = self.check_ssh_connection()
        results.append(("Verificar conexión SSH y Llama3", success))
        
        self.show_summary(results)
    
    def show_summary(self, results: list):
        """Mostrar resumen del diagnóstico"""
        print("\n============================================================")
        print("📋 RESUMEN DEL DIAGNÓSTICO")
        print("=" * 60)
        
        for test, success in results:
            status = "✅ EXITOSO" if success else "❌ FALLO"
            print(f"{status}: {test}")
        
        if not all(success for _, success in results):
            print("\n============================================================")
            print("💡 RECOMENDACIONES")
            print("=" * 60)
            print("- Verificar que todas las secciones estén presentes en ssh_credentials.json")
            print("- Verificar que las credenciales SSH sean correctas")
            print("- Verificar que el servidor tenga Llama3 instalado")
            print("- Verificar que haya suficiente memoria para ejecutar Llama3")

def main():
    diagnostic = SSHDiagnostic()
    diagnostic.run_diagnostic()

if __name__ == "__main__":
    main() 