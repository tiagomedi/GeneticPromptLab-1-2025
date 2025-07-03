#!/usr/bin/env python3
"""
Script de prueba para verificar el túnel SSH automático
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from setup_ssh_tunnel_auto import AutoSSHTunnel

def main():
    print("🧪 Prueba del túnel SSH automático")
    print("=" * 40)
    
    # Crear instancia del túnel
    tunnel = AutoSSHTunnel()
    
    try:
        # Configurar túnel
        if tunnel.setup_tunnel():
            print("✅ Túnel establecido correctamente")
            
            # Probar conexión
            if tunnel.test_connection():
                print("✅ Conexión funcionando")
            else:
                print("⚠️ Túnel activo pero Ollama no responde")
            
            # Mantener activo por 30 segundos
            print("⏳ Manteniendo túnel activo por 30 segundos...")
            import time
            time.sleep(30)
            
        else:
            print("❌ No se pudo establecer el túnel")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        tunnel.close_tunnel()
        print("🧹 Túnel cerrado")

if __name__ == "__main__":
    main() 