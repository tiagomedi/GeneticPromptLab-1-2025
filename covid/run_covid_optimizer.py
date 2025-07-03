#!/usr/bin/env python3
"""
Script principal para ejecutar el COVID-19 Genetic Prompt Optimizer
Automatiza la configuración del túnel SSH y la ejecución del optimizador
"""

import subprocess
import time
import sys
import os
import threading
import argparse
from typing import Optional

class CovidOptimizerRunner:
    """
    Gestor principal para ejecutar el optimizador de COVID-19
    """
    
    def __init__(self, 
                 ssh_port: int = 11435,
                 corpus_file: str = "corpus.csv",
                 model_name: str = "mistral",
                 population_size: int = 5,
                 generations: int = 3,
                 sample_size: int = 500):
        
        self.ssh_port = ssh_port
        self.corpus_file = corpus_file
        self.model_name = model_name
        self.population_size = population_size
        self.generations = generations
        self.sample_size = sample_size
        
        self.tunnel_process = None
        
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
            import requests
            import numpy
            print("✅ Dependencias verificadas")
        except ImportError as e:
            print(f"❌ Falta dependencia: {e}")
            print("   Ejecutar: pip install pandas scikit-learn requests numpy")
            return False
        
        return True
    
    def start_ssh_tunnel(self):
        """
        Inicia el túnel SSH en background
        """
        print("🔗 Iniciando túnel SSH...")
        
        try:
            # Comando para crear túnel SSH
            cmd = [
                'ssh', '-N', '-L', f'{self.ssh_port}:172.16.40.247:11434',
                'ignacio.medina1@200.14.84.16', '-p', '8080',
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'ServerAliveInterval=60',
                '-o', 'BatchMode=no'  # Permitir entrada de contraseña
            ]
            
            print(f"   Comando SSH: {' '.join(cmd)}")
            print("   💡 Se te pedirán las contraseñas:")
            print("      1. ignacio.udp2025")
            print("      2. research202x")
            
            # Crear túnel en background
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Esperar un poco para que se establezca
            print("   Esperando conexión...")
            time.sleep(10)
            
            # Verificar si el proceso sigue activo
            if self.tunnel_process.poll() is None:
                print("✅ Túnel SSH iniciado")
                return True
            else:
                print("❌ Error al iniciar túnel SSH")
                return False
                
        except Exception as e:
            print(f"❌ Error configurando túnel SSH: {e}")
            return False
    
    def test_connection(self):
        """
        Prueba la conexión con Ollama/Mistral
        """
        print("🧪 Probando conexión...")
        
        # Usar el script de prueba
        try:
            result = subprocess.run(
                [sys.executable, "test_connection.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Todas las pruebas pasaron")
                return True
            else:
                print("❌ Falló alguna prueba:")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Timeout en las pruebas")
            return False
        except Exception as e:
            print(f"❌ Error en pruebas: {e}")
            return False
    
    def run_optimizer(self):
        """
        Ejecuta el optimizador genético
        """
        print("🚀 Ejecutando optimizador genético...")
        
        # Configurar argumentos
        env = os.environ.copy()
        env['COVID_SSH_PORT'] = str(self.ssh_port)
        env['COVID_CORPUS_FILE'] = self.corpus_file
        env['COVID_MODEL_NAME'] = self.model_name
        env['COVID_POPULATION_SIZE'] = str(self.population_size)
        env['COVID_GENERATIONS'] = str(self.generations)
        env['COVID_SAMPLE_SIZE'] = str(self.sample_size)
        
        try:
            # Ejecutar optimizador
            result = subprocess.run(
                [sys.executable, "covid_genetic_optimizer.py"],
                env=env,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Optimización completada")
                return True
            else:
                print("❌ Error en optimización")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando optimizador: {e}")
            return False
    
    def cleanup(self):
        """
        Limpia recursos (cierra túnel SSH)
        """
        print("🧹 Limpiando recursos...")
        
        if self.tunnel_process:
            try:
                self.tunnel_process.terminate()
                self.tunnel_process.wait(timeout=5)
                print("✅ Túnel SSH cerrado")
            except subprocess.TimeoutExpired:
                self.tunnel_process.kill()
                print("⚠️ Túnel SSH forzado a cerrar")
            except Exception as e:
                print(f"⚠️ Error cerrando túnel: {e}")
    
    def run_full_pipeline(self):
        """
        Ejecuta el pipeline completo
        """
        print("🦠 COVID-19 Genetic Prompt Optimizer")
        print("=" * 50)
        
        try:
            # 1. Configurar entorno
            if not self.setup_environment():
                return False
            
            # 2. Iniciar túnel SSH
            if not self.start_ssh_tunnel():
                return False
            
            # 3. Probar conexión
            if not self.test_connection():
                print("⚠️ Falló la prueba de conexión, pero continuando...")
            
            # 4. Ejecutar optimizador
            success = self.run_optimizer()
            
            if success:
                print("\n🎉 ¡Pipeline completado exitosamente!")
                print("   Revisa los resultados en: covid_prompts.json")
            else:
                print("\n❌ Pipeline falló")
            
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
    Función principal con argumentos de línea de comandos
    """
    parser = argparse.ArgumentParser(description="COVID-19 Genetic Prompt Optimizer")
    
    parser.add_argument("--ssh-port", type=int, default=11435,
                       help="Puerto local para túnel SSH (default: 11435)")
    parser.add_argument("--corpus", type=str, default="corpus.csv",
                       help="Archivo del corpus (default: corpus.csv)")
    parser.add_argument("--model", type=str, default="mistral",
                       help="Modelo de LLM (default: mistral)")
    parser.add_argument("--population", type=int, default=5,
                       help="Tamaño de población inicial (default: 5)")
    parser.add_argument("--generations", type=int, default=3,
                       help="Número de generaciones (default: 3)")
    parser.add_argument("--sample-size", type=int, default=500,
                       help="Tamaño de muestra del corpus (default: 500)")
    parser.add_argument("--test-only", action="store_true",
                       help="Solo ejecutar pruebas de conexión")
    
    args = parser.parse_args()
    
    # Crear runner
    runner = CovidOptimizerRunner(
        ssh_port=args.ssh_port,
        corpus_file=args.corpus,
        model_name=args.model,
        population_size=args.population,
        generations=args.generations,
        sample_size=args.sample_size
    )
    
    if args.test_only:
        # Solo ejecutar pruebas
        print("🧪 Modo de prueba")
        runner.setup_environment()
        runner.start_ssh_tunnel()
        success = runner.test_connection()
        runner.cleanup()
    else:
        # Ejecutar pipeline completo
        success = runner.run_full_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 