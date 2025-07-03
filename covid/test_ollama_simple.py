#!/usr/bin/env python3
"""
Script simple para probar run_ollama_command
"""

from setup_ssh_tunnel_auto import RemoteSSHExecutor

def test_ollama_generation():
    """Probar la generación de texto con ollama"""
    print("🧪 Probando generación de texto con ollama...")
    
    executor = RemoteSSHExecutor()
    
    if not executor.connect():
        print("❌ No se pudo conectar")
        return False
    
    try:
        # Probar con un prompt muy simple
        simple_prompt = "Escribe una frase sobre el clima"
        print(f"📝 Probando prompt: {simple_prompt}")
        
        success, response = executor.run_ollama_command(simple_prompt)
        
        if success:
            print(f"✅ Respuesta recibida:")
            print(f"   {response}")
            print(f"   Longitud: {len(response)} caracteres")
            return True
        else:
            print(f"❌ Error: {response}")
            return False
            
    finally:
        executor.close()

if __name__ == "__main__":
    test_ollama_generation() 