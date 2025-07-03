#!/usr/bin/env python3
"""
Script simple para probar ollama directamente
"""

from setup_ssh_tunnel_auto import RemoteSSHExecutor
import time

def test_ollama_simple():
    """Probar ollama con comandos simples"""
    print("🧪 Probando ollama directamente...")
    
    executor = RemoteSSHExecutor()
    
    if not executor.connect():
        print("❌ No se pudo conectar")
        return False
    
    try:
        # Probar que ollama list funciona
        print("   📋 Verificando modelos disponibles...")
        success, output = executor.run_command("ollama list", timeout=10)
        if success:
            print(f"   ✅ Modelos disponibles:\n{output}")
        else:
            print(f"   ❌ Error: {output}")
            return False
        
        # Probar un comando muy simple
        print("   🧪 Probando comando simple...")
        success, output = executor.run_command("ollama run llama3.1 'Hi'", timeout=60)
        if success:
            print(f"   ✅ Respuesta:\n{output}")
            return True
        else:
            print(f"   ❌ Error: {output}")
            return False
            
    finally:
        executor.close()

if __name__ == "__main__":
    test_ollama_simple() 