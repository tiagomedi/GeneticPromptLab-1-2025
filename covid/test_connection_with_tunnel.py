#!/usr/bin/env python3
"""
Script integrado que activa el túnel SSH y ejecuta las pruebas de conexión
"""

import subprocess
import time
import sys
import os
import signal
import threading
from typing import Optional

# Importar el script de túnel automático
from setup_ssh_tunnel_auto import AutoSSHTunnel

class ConnectionTesterWithTunnel:
    """
    Tester de conexión que maneja automáticamente el túnel SSH
    """
    
    def __init__(self):
        self.tunnel = None
        self.tunnel_active = False
        
    def setup_tunnel(self) -> bool:
        """
        Configura el túnel SSH automáticamente
        """
        print("🔗 Configurando túnel SSH automático...")
        
        try:
            self.tunnel = AutoSSHTunnel()
            
            if self.tunnel.setup_tunnel():
                self.tunnel_active = True
                print("✅ Túnel SSH configurado exitosamente")
                
                # Esperar un poco más para asegurar la conexión
                print("⏳ Esperando estabilización del túnel...")
                time.sleep(5)
                
                # Verificar conexión básica
                if self.tunnel.test_connection():
                    print("✅ Túnel SSH verificado y funcionando")
                    return True
                else:
                    print("⚠️ Túnel activo pero Ollama no responde completamente")
                    return True  # Continuar de todos modos
            else:
                print("❌ No se pudo configurar el túnel SSH")
                return False
                
        except Exception as e:
            print(f"❌ Error configurando túnel: {e}")
            return False
    
    def run_tests(self) -> bool:
        """
        Ejecuta las pruebas de conexión
        """
        print("\n🧪 Ejecutando pruebas de conexión...")
        
        try:
            # Ejecutar script de pruebas
            result = subprocess.run(
                [sys.executable, "test_connection.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Mostrar salida
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("Errores:", result.stderr)
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("⏰ Timeout en las pruebas")
            return False
        except Exception as e:
            print(f"❌ Error ejecutando pruebas: {e}")
            return False
    
    def cleanup(self):
        """
        Limpia recursos y cierra el túnel
        """
        if self.tunnel and self.tunnel_active:
            print("\n🧹 Cerrando túnel SSH...")
            self.tunnel.close_tunnel()
            self.tunnel_active = False
            print("✅ Túnel SSH cerrado")
    
    def run_full_test(self) -> bool:
        """
        Ejecuta el ciclo completo: túnel + pruebas + limpieza
        """
        print("🚀 COVID-19 Genetic Prompt Lab - Pruebas Completas")
        print("=" * 60)
        
        try:
            # 1. Configurar túnel SSH
            if not self.setup_tunnel():
                return False
            
            # 2. Ejecutar pruebas
            success = self.run_tests()
            
            # 3. Mostrar resultado
            if success:
                print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
                print("   El sistema está listo para usar.")
                print("\n📋 Próximos pasos:")
                print("   1. Ejecutar: python3 covid_genetic_optimizer.py")
                print("   2. O usar: python3 run_covid_optimizer.py")
            else:
                print("\n❌ Algunas pruebas fallaron")
                print("   Revisa los errores anteriores")
            
            return success
            
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por el usuario")
            return False
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """
    Función principal
    """
    # Manejar señales de terminación
    def signal_handler(signum, frame):
        print("\n🛑 Terminando...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Crear tester y ejecutar
    tester = ConnectionTesterWithTunnel()
    success = tester.run_full_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 