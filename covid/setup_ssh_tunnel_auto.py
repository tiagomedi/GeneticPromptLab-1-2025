#!/usr/bin/env python3
"""
Script para configurar túnel SSH automáticamente usando credenciales del archivo JSON
"""

import json
import subprocess
import time
import os
import signal
import sys
import threading
from typing import Optional

try:
    import pexpect
except ImportError:
    print("❌ pexpect no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pexpect"])
    import pexpect

class AutoSSHTunnel:
    def __init__(self, credentials_file: str = "ssh_credentials.json"):
        self.credentials_file = credentials_file
        self.tunnel_process: Optional[pexpect.spawn] = None
        self.config = self._load_credentials()
        
    def _load_credentials(self) -> dict:
        """Cargar credenciales desde archivo JSON"""
        try:
            with open(self.credentials_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Archivo de credenciales no encontrado: {self.credentials_file}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Error al leer el archivo JSON: {self.credentials_file}")
            sys.exit(1)
    
    def setup_tunnel(self) -> bool:
        """Configurar túnel SSH automáticamente"""
        print("🔗 Configurando túnel SSH automático...")
        
        jump_host = self.config["ssh_config"]["jump_host"]
        target_host = self.config["ssh_config"]["target_host"]
        tunnel_config = self.config["tunnel_config"]
        
        # Comando SSH con ProxyJump
        ssh_command = (
            f"ssh -N -L {tunnel_config['local_port']}:{tunnel_config['remote_host']}:{tunnel_config['remote_port']} "
            f"-J {jump_host['username']}@{jump_host['host']}:{jump_host['port']} "
            f"{target_host['username']}@{target_host['host']} "
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-o ServerAliveInterval=60"
        )
        
        print(f"   Ejecutando: {ssh_command}")
        print("   🔑 Usando credenciales automáticas...")
        
        try:
            # Iniciar proceso SSH
            self.tunnel_process = pexpect.spawn(ssh_command, timeout=30)
            
            # Manejar primera contraseña (jump host)
            index = self.tunnel_process.expect([
                "password:",
                "Password:",
                pexpect.TIMEOUT,
                pexpect.EOF
            ])
            
            if index in [0, 1]:
                print("   🔑 Enviando primera contraseña (jump host)...")
                self.tunnel_process.sendline(jump_host["password"])
                
                # Manejar segunda contraseña (target host)
                index = self.tunnel_process.expect([
                    "password:",
                    "Password:",
                    pexpect.TIMEOUT,
                    pexpect.EOF
                ])
                
                if index in [0, 1]:
                    print("   🔑 Enviando segunda contraseña (target host)...")
                    self.tunnel_process.sendline(target_host["password"])
                    
                    # Esperar a que se establezca la conexión
                    time.sleep(3)
                    
                    if self.tunnel_process.isalive():
                        print("✅ Túnel SSH establecido exitosamente")
                        return True
                    else:
                        print("❌ El túnel SSH se cerró inesperadamente")
                        return False
                else:
                    print("❌ Timeout o error esperando segunda contraseña")
                    return False
            else:
                print("❌ Timeout o error esperando primera contraseña")
                return False
                
        except pexpect.exceptions.TIMEOUT:
            print("❌ Timeout al establecer túnel SSH")
            return False
        except Exception as e:
            print(f"❌ Error al establecer túnel SSH: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Probar la conexión del túnel"""
        import requests
        
        local_port = self.config["tunnel_config"]["local_port"]
        test_url = f"http://localhost:{local_port}/api/tags"
        
        try:
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                print("✅ Túnel SSH funcionando correctamente")
                return True
            else:
                print(f"❌ Error en túnel SSH: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error probando túnel SSH: {e}")
            return False
    
    def close_tunnel(self):
        """Cerrar el túnel SSH"""
        if self.tunnel_process and self.tunnel_process.isalive():
            print("🧹 Cerrando túnel SSH...")
            self.tunnel_process.terminate()
            self.tunnel_process.wait()
            print("✅ Túnel SSH cerrado")
    
    def keep_alive(self):
        """Mantener el túnel activo"""
        def signal_handler(signum, frame):
            print("\n🛑 Recibida señal de terminación...")
            self.close_tunnel()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("⏳ Manteniendo túnel activo... (Ctrl+C para terminar)")
        try:
            while self.tunnel_process and self.tunnel_process.isalive():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Terminando túnel SSH...")
            self.close_tunnel()

def main():
    print("🚀 Configurador Automático de Túnel SSH")
    print("=" * 50)
    
    tunnel = AutoSSHTunnel()
    
    try:
        if tunnel.setup_tunnel():
            # Probar conexión
            print("\n🧪 Probando conexión...")
            if tunnel.test_connection():
                print("\n🎉 ¡Túnel SSH configurado y funcionando!")
                print("   El túnel estará activo hasta que cierres este script.")
                tunnel.keep_alive()
            else:
                print("\n⚠️ Túnel establecido pero no responde. Verifica Ollama en el servidor.")
                tunnel.keep_alive()
        else:
            print("\n❌ No se pudo establecer el túnel SSH")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        tunnel.close_tunnel()
        sys.exit(1)

if __name__ == "__main__":
    main() 