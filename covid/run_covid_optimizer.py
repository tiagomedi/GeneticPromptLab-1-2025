#!/usr/bin/env python3
"""
Script principal para ejecutar el COVID-19 Genetic Prompt Optimizer
Ejecuta Llama3 directamente en el servidor remoto
"""

import subprocess
import time
import sys
import os
import argparse
from typing import Optional

class CovidOptimizerRunner:
    """
    Gestor principal para ejecutar el optimizador de COVID-19
    """
    
    def __init__(self, 
                 corpus_file: str = "corpus.csv",
                 population_size: int = 5,
                 generations: int = 3,
                 mutation_rate: float = 0.1,
                 sample_size: int = 500):
        
        self.corpus_file = corpus_file
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.sample_size = sample_size
        
    def setup_environment(self):
        """
        Configura el entorno necesario
        """
        print("🔧 Configurando entorno...")
        
        # Verificar que estamos en el directorio correcto
        if not os.path.exists(self.corpus_file):
            print(f"❌ No se encuentra {self.corpus_file}")
            print("   Asegúrate de estar en el directorio /covid")
            return False
        
        # Verificar dependencias básicas
        try:
            import pandas
            import sklearn
            import pexpect
            import numpy
            print("✅ Dependencias verificadas")
        except ImportError as e:
            print(f"❌ Falta dependencia: {e}")
            print("   Ejecutar: pip install pandas scikit-learn pexpect numpy")
            return False
        
        return True
    
    def test_connection(self):
        """
        Prueba la conexión con el servidor remoto y Llama3
        """
        print("🧪 Probando conexión...")
        
        from setup_ssh_tunnel_auto import RemoteSSHExecutor
        
        try:
            executor = RemoteSSHExecutor()
            
            if executor.connect():
                if executor.test_llama():
                    print("✅ Conexión y Llama3 verificados")
                    executor.close()
                    return True
                else:
                    print("❌ Llama3 no está funcionando")
                    executor.close()
                    return False
            else:
                print("❌ No se pudo conectar al servidor")
                return False
                
        except Exception as e:
            print(f"❌ Error probando conexión: {e}")
            return False
    
    def run_optimizer(self):
        """
        Ejecuta el optimizador genético
        """
        print("🚀 Ejecutando optimizador genético...")
        
        try:
            from covid_genetic_optimizer import CovidGeneticOptimizer
            
            optimizer = CovidGeneticOptimizer(
                corpus_file=self.corpus_file,
                population_size=self.population_size,
                generations=self.generations,
                mutation_rate=self.mutation_rate,
                sample_size=self.sample_size
            )
            
            optimizer.optimize()
            return True
            
        except Exception as e:
            print(f"❌ Error ejecutando optimizador: {e}")
            return False
    
    def run_full_pipeline(self):
        """
        Ejecuta el pipeline completo
        """
        print("🦠 COVID-19 Genetic Prompt Optimizer")
        print("=" * 50)
        self._show_system_info()
        
        try:
            # 1. Configurar entorno
            if not self.setup_environment():
                return False
            
            # 2. Probar conexión
            if not self.test_connection():
                print("❌ Falló la prueba de conexión")
                return False
            
            # 3. Ejecutar optimizador
            success = self.run_optimizer()
            
            if success:
                print("\n🎉 ¡Pipeline completado exitosamente!")
                print("   Revisa los resultados en: covid_prompts.json")
                self._show_completion_info()
            else:
                print("\n❌ Pipeline falló")
                self._show_troubleshooting_info()
            
            return success
            
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por el usuario")
            return False
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            return False
    
    def _show_system_info(self):
        """
        Muestra información del sistema al inicio
        """
        print("📋 Configuración:")
        print(f"   📁 Corpus: {self.corpus_file}")
        print(f"   👥 Población: {self.population_size}")
        print(f"   🧬 Generaciones: {self.generations}")
        print(f"   🎲 Tasa de mutación: {self.mutation_rate}")
        print(f"   📏 Tamaño de muestra: {self.sample_size}")
        print()
    
    def _show_completion_info(self):
        """
        Muestra información de completado exitoso
        """
        print("\n📊 Información del procesamiento:")
        print(f"   📁 Corpus: {self.corpus_file}")
        print(f"   👥 Población: {self.population_size}")
        print(f"   🧬 Generaciones: {self.generations}")
        print(f"   🎲 Tasa de mutación: {self.mutation_rate}")
        print(f"   📏 Muestra: {self.sample_size}")
        
        print("\n📋 Archivos generados:")
        if os.path.exists("covid_prompts.json"):
            print("   ✅ covid_prompts.json - Prompts optimizados")
        
        print("\n🔄 Próximos pasos:")
        print("   1. Revisar prompts generados en covid_prompts.json")
        print("   2. Analizar la efectividad de los prompts")
        print("   3. Ejecutar nuevamente con diferentes parámetros si es necesario")
    
    def _show_troubleshooting_info(self):
        """
        Muestra información de solución de problemas
        """
        print("\n🔧 Solución de problemas:")
        print("   1. Verificar conexión a internet")
        print("   2. Verificar credenciales en ssh_credentials.json")
        print("   3. Verificar que el servidor remoto esté accesible")
        print("   4. Verificar que Llama3 esté instalado y funcionando")
        
        print("\n🚀 Comandos útiles:")
        print("   - Probar conexión: python3 diagnostico_ssh.py")
        print("   - Verificar Llama3: ollama list")
        
        print("\n📞 Diagnóstico:")
        print("   - Verificar que las credenciales SSH sean correctas")
        print("   - Verificar que Llama3 esté instalado en el servidor")
        print("   - Verificar que el servidor tenga suficiente memoria para Llama3")

def main():
    """
    Función principal con argumentos de línea de comandos
    """
    parser = argparse.ArgumentParser(description="COVID-19 Genetic Prompt Optimizer")
    
    parser.add_argument("--corpus", type=str, default="corpus.csv",
                       help="Archivo del corpus (default: corpus.csv)")
    parser.add_argument("--population", type=int, default=5,
                       help="Tamaño de población inicial (default: 5)")
    parser.add_argument("--generations", type=int, default=3,
                       help="Número de generaciones (default: 3)")
    parser.add_argument("--mutation-rate", type=float, default=0.1,
                       help="Tasa de mutación (default: 0.1)")
    parser.add_argument("--sample-size", type=int, default=500,
                       help="Tamaño de muestra del corpus (default: 500)")
    parser.add_argument("--test-only", action="store_true",
                       help="Solo ejecutar pruebas de conexión")
    
    args = parser.parse_args()
    
    # Crear runner
    runner = CovidOptimizerRunner(
        corpus_file=args.corpus,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        sample_size=args.sample_size
    )
    
    if args.test_only:
        # Solo ejecutar pruebas
        print("🧪 Modo de prueba")
        success = runner.test_connection()
    else:
        # Ejecutar pipeline completo
        success = runner.run_full_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 