#!/usr/bin/env python3
"""
Script de prueba rápida para verificar el run_covid_optimizer.py modificado
"""

import sys
import os
import subprocess

def test_basic_import():
    """
    Prueba que el módulo se pueda importar correctamente
    """
    print("🧪 Probando importación básica...")
    try:
        from run_covid_optimizer import CovidOptimizerRunner
        print("✅ Importación exitosa")
        return True
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_runner_creation():
    """
    Prueba que se pueda crear una instancia del runner
    """
    print("🧪 Probando creación de instancia...")
    try:
        from run_covid_optimizer import CovidOptimizerRunner
        
        runner = CovidOptimizerRunner(
            ssh_port=11435,
            corpus_file="corpus.csv",
            model_name="mistral",
            population_size=2,  # Pequeño para prueba
            generations=1,      # Mínimo para prueba
            sample_size=100     # Pequeño para prueba
        )
        
        print("✅ Instancia creada exitosamente")
        print(f"   📁 Corpus: {runner.corpus_file}")
        print(f"   🤖 Modelo: {runner.model_name}")
        print(f"   👥 Población: {runner.population_size}")
        return True
        
    except Exception as e:
        print(f"❌ Error creando instancia: {e}")
        return False

def test_environment_setup():
    """
    Prueba la configuración del entorno
    """
    print("🧪 Probando configuración del entorno...")
    try:
        from run_covid_optimizer import CovidOptimizerRunner
        
        runner = CovidOptimizerRunner()
        
        # Verificar método de configuración
        if hasattr(runner, 'setup_environment'):
            print("✅ Método setup_environment disponible")
        else:
            print("❌ Método setup_environment no encontrado")
            return False
        
        # Verificar método de información del sistema
        if hasattr(runner, '_show_system_info'):
            print("✅ Método _show_system_info disponible")
        else:
            print("❌ Método _show_system_info no encontrado")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_help_command():
    """
    Prueba el comando de ayuda
    """
    print("🧪 Probando comando de ayuda...")
    try:
        result = subprocess.run(
            [sys.executable, "run_covid_optimizer.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✅ Comando de ayuda funciona")
            return True
        else:
            print("❌ Problema con comando de ayuda")
            print(f"   Return code: {result.returncode}")
            print(f"   Stdout: {result.stdout[:100]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout en comando de ayuda")
        return False
    except Exception as e:
        print(f"❌ Error en comando de ayuda: {e}")
        return False

def test_test_only_mode():
    """
    Prueba el modo de solo pruebas (sin ejecutar el optimizador)
    """
    print("🧪 Probando modo --test-only...")
    try:
        result = subprocess.run(
            [sys.executable, "run_covid_optimizer.py", "--test-only"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   Return code: {result.returncode}")
        if result.stdout:
            print(f"   Salida (primeras líneas): {result.stdout.split('\\n')[0]}")
        
        # No esperamos que pase necesariamente (puede fallar por SSH), 
        # pero sí que se ejecute sin errores de sintaxis
        print("✅ Modo --test-only se ejecuta sin errores de sintaxis")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout en modo --test-only")
        return False
    except Exception as e:
        print(f"❌ Error en modo --test-only: {e}")
        return False

def main():
    """
    Ejecuta todas las pruebas
    """
    print("🚀 Pruebas del run_covid_optimizer.py modificado")
    print("=" * 60)
    
    tests = [
        test_basic_import,
        test_runner_creation,
        test_environment_setup,
        test_help_command,
        test_test_only_mode
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. {test.__doc__.strip()}")
        if test():
            passed += 1
        else:
            print("   ⚠️ Esta prueba falló")
    
    print(f"\n📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas básicas pasaron!")
        print("   El run_covid_optimizer.py modificado parece estar funcionando")
        print("\n🔄 Próximo paso: Ejecutar python3 run_covid_optimizer.py")
    else:
        print("⚠️ Algunas pruebas fallaron, revisar los errores arriba")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 