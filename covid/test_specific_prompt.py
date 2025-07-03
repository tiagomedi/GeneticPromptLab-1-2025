#!/usr/bin/env python3
"""
Script para probar el prompt específico que está fallando
"""

from setup_ssh_tunnel_auto import RemoteSSHExecutor

def test_specific_prompt():
    """Probar el prompt específico que está fallando"""
    print("🧪 Probando prompt específico...")
    
    executor = RemoteSSHExecutor()
    
    if not executor.connect():
        print("❌ No se pudo conectar")
        return False
    
    try:
        # Probar el prompt que está fallando
        failing_prompt = "Escribe un texto similar a: corona patient number cross 400 in india india is in"
        print(f"📝 Probando prompt: {failing_prompt}")
        
        success, response = executor.run_ollama_command(failing_prompt)
        
        print(f"✅ Success: {success}")
        print(f"📝 Respuesta completa:")
        print(f"'{response}'")
        print(f"📏 Longitud: {len(response)} caracteres")
        
        # También probar un prompt más simple
        simple_prompt = "Escribe sobre coronavirus"
        print(f"\n📝 Probando prompt simple: {simple_prompt}")
        
        success2, response2 = executor.run_ollama_command(simple_prompt)
        
        print(f"✅ Success: {success2}")
        print(f"📝 Respuesta completa:")
        print(f"'{response2}'")
        print(f"📏 Longitud: {len(response2)} caracteres")
            
    finally:
        executor.close()

if __name__ == "__main__":
    test_specific_prompt() 