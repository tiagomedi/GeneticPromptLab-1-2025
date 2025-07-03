#!/usr/bin/env python3
"""
Script de prueba para verificar conexión SSH y Mistral
"""

import requests
import json
import time
import sys
import pandas as pd
import os
import pexpect
import subprocess


def test_ssh_tunnel(port=11435):
    """
    Prueba la conexión SSH tunnel
    """
    print(f"🔗 Probando túnel SSH en puerto {port}...")
    
    try:
        # Probar conexión básica
        response = requests.get(f'http://localhost:{port}/api/tags', timeout=5)
        
        if response.status_code == 200:
            print("✅ Túnel SSH funcionando correctamente")
            data = response.json()
            models = data.get('models', [])
            print(f"   Modelos disponibles: {len(models)}")
            
            for model in models:
                print(f"   - {model.get('name', 'Unknown')}")
            
            return True
        else:
            print(f"❌ Error en túnel SSH: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ No se puede conectar al túnel SSH: {e}")
        print("   Asegúrate de que el túnel SSH esté activo:")
        print("   python setup_ssh_tunnel_auto.py")
        return False

def test_mistral_connection(port=11435, model_name="mistral"):
    """
    Prueba la conexión específica con Mistral
    """
    print(f"🤖 Probando conexión con {model_name}...")
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond briefly."
            },
            {
                "role": "user",
                "content": "Hello! Can you respond with just 'Connection successful'?"
            }
        ],
        "stream": False,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f'http://localhost:{port}/api/chat', 
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("message", {}).get("content", "")
            print(f"✅ Mistral respondió: {content[:100]}...")
            return True
        else:
            print(f"❌ Error en Mistral: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error conectando con Mistral: {e}")
        return False

def test_corpus_loading(corpus_file="corpus.csv"):
    """
    Prueba la carga del archivo corpus
    """
    print(f"📁 Probando carga del corpus: {corpus_file}")
    
    try:
        # Verificar que el archivo existe
        if not os.path.exists(corpus_file):
            print(f"❌ Archivo no encontrado: {corpus_file}")
            return False
        
        # Obtener tamaño del archivo
        file_size = os.path.getsize(corpus_file) / (1024 * 1024)  # MB
        print(f"   Tamaño del archivo: {file_size:.1f} MB")
        
        # Leer las primeras filas
        print("   Leyendo primeras filas...")
        df = pd.read_csv(corpus_file, nrows=5)
        
        print(f"   Columnas: {list(df.columns)}")
        print(f"   Número de filas (muestra): {len(df)}")
        
        # Detectar columna de texto principal
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                text_columns.append((col, avg_length))
        
        if text_columns:
            main_col, avg_len = max(text_columns, key=lambda x: x[1])
            print(f"   Columna de texto principal: {main_col} (promedio: {avg_len:.0f} chars)")
            
            # Mostrar ejemplo de texto
            example_text = str(df[main_col].iloc[0])
            print(f"   Ejemplo de texto: {example_text[:200]}...")
        
        print("✅ Corpus cargado correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error cargando corpus: {e}")
        return False

def test_dependencies():
    """
    Verifica que todas las dependencias estén instaladas
    """
    print("📦 Verificando dependencias...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NO INSTALADO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Faltan dependencias: {', '.join(missing_packages)}")
        print("   Instalar con: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def main():
    """
    Ejecuta todas las pruebas
    """
    print("🧪 COVID-19 Genetic Prompt Lab - Pruebas de Conexión")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Dependencias
    print("\n1. 📦 Verificando dependencias...")
    if not test_dependencies():
        all_tests_passed = False
    
    # Test 2: Corpus
    print("\n2. 📁 Verificando corpus...")
    if not test_corpus_loading():
        all_tests_passed = False
    
    # Test 3: SSH Tunnel
    print("\n3. 🔗 Verificando túnel SSH...")
    if not test_ssh_tunnel():
        all_tests_passed = False
    
    # Test 4: Mistral
    print("\n4. 🤖 Verificando Mistral...")
    if not test_mistral_connection():
        all_tests_passed = False
    
    # Resultado final
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ¡Todas las pruebas pasaron! El sistema está listo.")
        print("\nPuedes ejecutar:")
        print("   python covid_genetic_optimizer.py")
    else:
        print("❌ Algunas pruebas fallaron. Revisa los errores arriba.")
        print("\nPasos para solucionar:")
        print("1. Instalar dependencias faltantes")
        print("2. Verificar que corpus.csv esté presente")
        print("3. Configurar túnel SSH: python setup_ssh_tunnel_auto.py")
        print("4. Verificar que Mistral esté corriendo en el servidor")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 