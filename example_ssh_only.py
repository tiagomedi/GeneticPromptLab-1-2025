#!/usr/bin/env python3
"""
Ejemplo de uso del sistema GeneticPromptLab SOLO con SSH/Ollama
Sistema actualizado que elimina completamente OpenAI
"""

import sys
import os
import pandas as pd
import numpy as np
from GeneticPromptLab.qa_optim import QuestionsAnswersOptimizer
from ssh_connection import LLMRemoteExecutor

def test_ssh_connection():
    """
    Probar conexión SSH antes de comenzar
    """
    print("🔗 Probando conexión SSH...")
    try:
        ssh_executor = LLMRemoteExecutor("ssh_credentials.json")
        success = ssh_executor.test_setup()
        
        if success:
            print("✅ Conexión SSH funcionando correctamente")
            return True
        else:
            print("❌ Problema con conexión SSH o LLM")
            return False
    except Exception as e:
        print(f"❌ Error de conexión SSH: {e}")
        return False

def load_sample_data():
    """
    Cargar datos de ejemplo para clasificación
    """
    print("📄 Cargando datos de ejemplo...")
    
    # Datos de ejemplo para clasificación de noticias
    sample_questions = [
        "Apple reported record quarterly earnings driven by strong iPhone sales",
        "The soccer world cup final will be held next month in Qatar",
        "Scientists discover new exoplanet that might support life",
        "Tesla stock price surged after announcing new factory expansion",
        "Olympic swimming records were broken at the latest championship",
        "New AI breakthrough shows promise in medical diagnosis",
        "Microsoft acquires gaming company for $70 billion deal",
        "Basketball season opens with exciting matchups this weekend",
        "Climate change study reveals alarming temperature increases",
        "Amazon reports strong holiday shopping season results"
    ]
    
    # Labels: 0=Business, 1=Sports, 2=Science, 3=Technology
    sample_labels = [0, 1, 2, 0, 1, 3, 0, 1, 2, 0]
    
    label_dict = {
        0: "Business",
        1: "Sports", 
        2: "Science",
        3: "Technology"
    }
    
    print(f"✅ Cargados {len(sample_questions)} ejemplos de clasificación")
    return sample_questions, sample_labels, label_dict

def run_genetic_optimization():
    """
    Ejecutar optimización genética completa
    """
    print("\n🧬 Iniciando optimización genética...")
    
    # Cargar datos
    questions, labels, label_dict = load_sample_data()
    
    # Dividir en train/test
    train_size = int(0.8 * len(questions))
    train_questions = questions[:train_size]
    train_labels = labels[:train_size]
    test_questions = questions[train_size:]
    test_labels = labels[train_size:]
    
    print(f"📊 Train: {len(train_questions)} ejemplos, Test: {len(test_questions)} ejemplos")
    
    # Configurar optimizador
    optimizer = QuestionsAnswersOptimizer(
        problem_description="Classify news articles into Business, Sports, Science, or Technology categories",
        train_questions_list=train_questions,
        train_answers_label=train_labels,
        test_questions_list=test_questions,
        test_answers_label=test_labels,
        label_dict=label_dict,
        model_name="all-MiniLM-L6-v2",
        sample_p=1.0,
        init_and_fitness_sample=5,  # Pequeño para pruebas
        window_size_init=1,
        generations=3,  # Pocas generaciones para pruebas
        num_retries=1,
        ssh_credentials="ssh_credentials.json",
        modelo_llm="llama3.1",
        temperatura=0.7
    )
    
    print(f"✅ Optimizador configurado")
    
    # Ejecutar algoritmo genético
    print(f"\n🚀 Ejecutando algoritmo genético...")
    final_population = optimizer.genetic_algorithm(mutation_rate=0.2)
    
    print(f"\n🎉 Optimización completada!")
    print(f"📝 Población final:")
    for i, prompt in enumerate(final_population[:3]):  # Mostrar solo los primeros 3
        print(f"  {i+1}. {prompt[:100]}...")
    
    return final_population

def test_individual_prompt():
    """
    Probar un prompt individual
    """
    print("\n🧪 Probando prompt individual...")
    
    # Datos de prueba
    questions, labels, label_dict = load_sample_data()
    
    # Configurar optimizador
    optimizer = QuestionsAnswersOptimizer(
        problem_description="Classify news articles into categories",
        train_questions_list=questions[:5],
        train_answers_label=labels[:5],
        label_dict=label_dict,
        init_and_fitness_sample=2,
        ssh_credentials="ssh_credentials.json",
        modelo_llm="llama3.1"
    )
    
    # Generar un prompt
    prompts = optimizer.generate_init_prompts(1)
    
    if prompts:
        test_prompt = prompts[0]
        print(f"📝 Prompt generado: {test_prompt}")
        
        # Evaluar fitness
        fitness_scores = optimizer.evaluate_fitness([test_prompt])
        print(f"📊 Fitness score: {fitness_scores[0]:.3f}")
    else:
        print("❌ No se pudo generar prompt")

def main():
    """
    Función principal del ejemplo
    """
    print("🚀 Sistema GeneticPromptLab - Solo SSH/Ollama")
    print("=" * 60)
    
    # Verificar conexión SSH
    if not test_ssh_connection():
        print("\n❌ No se puede continuar sin conexión SSH")
        print("💡 Asegúrate de que:")
        print("   - ssh_credentials.json esté configurado correctamente")
        print("   - El servidor remoto esté accesible")
        print("   - Ollama esté ejecutándose en el servidor")
        return False
    
    # Menú de opciones
    print(f"\n🎯 Opciones disponibles:")
    print("1. Ejecutar optimización genética completa")
    print("2. Probar un prompt individual")
    print("3. Solo verificar que todo funciona")
    print("4. Salir")
    
    try:
        choice = input("\nSelecciona opción (1-4): ").strip()
        
        if choice == '1':
            final_population = run_genetic_optimization()
            print(f"\n💾 Resultados guardados en directorio 'runs/'")
            
        elif choice == '2':
            test_individual_prompt()
            
        elif choice == '3':
            print("✅ Verificación completada - Sistema funcionando correctamente")
            
        elif choice == '4':
            print("👋 ¡Hasta luego!")
            return True
            
        else:
            print("❌ Opción no válida")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Operación cancelada por el usuario")
        return False
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 