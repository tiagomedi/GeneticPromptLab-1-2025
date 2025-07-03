#!/usr/bin/env python3
"""
Script para configurar túnel SSH en cadena para acceder a Mistral/Ollama
"""

import subprocess
import time
import threading
import os
import sys

class SSHTunnelManager:
    def __init__(self):
        self.tunnel_process = None
        self.local_port = 11435  # Puerto local para el túnel
        
    def create_tunnel(self):
        """
        Crea un túnel SSH en cadena:
        Local:11435 -> 200.14.84.16:8080 -> 172.16.40.247:11434
        """
        try:
            # Comando para crear túnel SSH en cadena
            # Primero conecta al servidor intermedio, luego al servidor final
            cmd = [
                'ssh', '-N', '-L', f'{self.local_port}:172.16.40.247:11434',
                'ignacio.medina1@200.14.84.16', '-p', '8080',
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'ServerAliveInterval=60'
            ]
            
            print(f"🔗 Creando túnel SSH...")
            print(f"   Local: localhost:{self.local_port}")
            print(f"   Remoto: 172.16.40.247:11434 (vía 200.14.84.16:8080)")
            print(f"   Comando: {' '.join(cmd)}")
            
            # Crear el túnel
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Esperar un poco para que se establezca la conexión
            time.sleep(3)
            
            # Verificar si el túnel está activo
            if self.tunnel_process.poll() is None:
                print("✅ Túnel SSH establecido correctamente")
                return True
            else:
                print("❌ Error al establecer el túnel SSH")
                return False
                
        except Exception as e:
            print(f"❌ Error al crear túnel SSH: {e}")
            return False
    
    def test_connection(self):
        """
        Prueba la conexión con Ollama/Mistral a través del túnel
        """
        import requests
        
        try:
            # Probar conexión a Ollama
            response = requests.get(f'http://localhost:{self.local_port}/api/tags', timeout=10)
            if response.status_code == 200:
                print("✅ Conexión con Ollama exitosa")
                models = response.json().get('models', [])
                print(f"   Modelos disponibles: {[m['name'] for m in models]}")
                return True
            else:
                print(f"❌ Error en conexión: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Error al conectar con Ollama: {e}")
            return False
    
    def close_tunnel(self):
        """
        Cierra el túnel SSH
        """
        if self.tunnel_process:
            print("🔒 Cerrando túnel SSH...")
            self.tunnel_process.terminate()
            self.tunnel_process.wait()
            print("✅ Túnel SSH cerrado")

def main():
    print("🚀 Configurando túnel SSH para Mistral/Ollama")
    print("=" * 50)
    
    # Verificar que SSH esté disponible
    try:
        subprocess.run(['ssh', '-V'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ SSH no está disponible en el sistema")
        sys.exit(1)
    
    # Crear y probar túnel
    tunnel_manager = SSHTunnelManager()
    
    if tunnel_manager.create_tunnel():
        time.sleep(5)  # Esperar más tiempo para estabilizar conexión
        
        if tunnel_manager.test_connection():
            print("\n✅ Configuración completa!")
            print(f"   Puedes usar: http://localhost:{tunnel_manager.local_port}")
            print("   Presiona Ctrl+C para cerrar el túnel")
            
            try:
                # Mantener el túnel activo
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Interrupción detectada")
        else:
            print("\n❌ No se pudo establecer conexión con Ollama")
    else:
        print("\n❌ No se pudo crear el túnel SSH")
    
    tunnel_manager.close_tunnel()

if __name__ == "__main__":
    main() 