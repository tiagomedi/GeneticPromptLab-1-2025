#!/usr/bin/env python3
"""
Ejemplo de uso del sistema GeneticPromptLab para detección COVID
Usando estructura: Role + Task Description + Modifics
"""

import sys
import os
import logging
from GeneticPromptLab.base_class import GeneticPromptLab

def main():
    """
    Ejemplo completo de optimización genética para prompts COVID
    """
    print("🚀 GeneticPromptLab - Sistema de Detección COVID")
    print("=" * 60)
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Inicializar optimizador
        print("\n🔧 Configurando optimizador genético...")
        
        optimizer = GeneticPromptLab(
            corpus_file='data/corpus.csv',      # Usar corpus principal
            population_size=8,                  # Población pequeña para pruebas
            generations=3,                      # Pocas generaciones para pruebas
            mutation_rate=0.3,                  # Tasa de mutación alta
            sample_size=500,                    # Limitar corpus para velocidad
            modelo_llm='llama3.1',             # Modelo Ollama
            temperatura=0.7,                    # Temperatura para creatividad
            credentials_file='ssh_credentials.json'  # Credenciales SSH
        )
        
        print(f"✅ Optimizador configurado - Corpus: {len(optimizer.corpus_data)} textos")
        
        # Verificar conexión SSH
        print("\n🔗 Verificando conexión SSH...")
        try:
            ssh_status = optimizer.ssh_executor.test_setup()
            if ssh_status:
                print("✅ Conexión SSH y LLM funcionando correctamente")
            else:
                print("❌ Problema con conexión SSH o LLM")
                print("💡 Verifica ssh_credentials.json y que Ollama esté ejecutándose")
                return False
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False
        
        # Mostrar ejemplos de estructura de prompt
        print("\n📋 Ejemplos de componentes de prompt:")
        print("Roles disponibles:")
        for i, role in enumerate(optimizer.roles[:5]):
            print(f"  {i+1}. {role}")
        
        print("\nTasks disponibles:")
        for i, task in enumerate(optimizer.task_descriptions[:3]):
            print(f"  {i+1}. {task}")
        
        print("\nModifics disponibles:")
        for i, modific in enumerate(optimizer.modifics_pool[:5]):
            print(f"  {i+1}. {modific}")
        
        # Generar ejemplo de prompt
        print("\n📝 Ejemplo de prompt estructurado:")
        sample_texts = optimizer.sample_distinct(1)
        sample_prompts = optimizer.create_prompts(sample_texts)
        
        if sample_prompts:
            example = sample_prompts[0]
            print(f"Role: {example['role']}")
            print(f"Task: {example['task_description']}")
            print(f"Modifics: {example['modifics']}")
            print(f"Texto: {example['texto'][:100]}...")
            print(f"\nPrompt completo:")
            print("-" * 40)
            print(example['full_prompt'])
            print("-" * 40)
        
        # Preguntar al usuario qué hacer
        print("\n🎯 Opciones disponibles:")
        print("1. Ejecutar algoritmo genético completo")
        print("2. Evaluar un prompt específico")
        print("3. Generar datos de ejemplo")
        print("4. Salir")
        
        choice = input("\nSelecciona opción (1-4): ").strip()
        
        if choice == '1':
            ejecutar_algoritmo_completo(optimizer)
        elif choice == '2':
            evaluar_prompt_especifico(optimizer)
        elif choice == '3':
            generar_datos_ejemplo(optimizer)
        elif choice == '4':
            print("👋 ¡Hasta luego!")
            return True
        else:
            print("❌ Opción no válida")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error en ejemplo: {e}")
        return False

def ejecutar_algoritmo_completo(optimizer):
    """
    Ejecutar algoritmo genético completo
    """
    print("\n🧬 Ejecutando algoritmo genético completo...")
    print("⏳ Esto puede tomar varios minutos...")
    
    try:
        # Ejecutar algoritmo
        results = optimizer.genetic_algorithm()
        
        # Mostrar resultados
        print("\n🎉 Algoritmo genético completado!")
        print("=" * 50)
        
        best_prompt = results['best_prompt']
        print(f"🏆 Mejor prompt encontrado:")
        print(f"  Role: {best_prompt['role']}")
        print(f"  Task: {best_prompt['task_description']}")
        print(f"  Modifics: {best_prompt['modifics']}")
        print(f"  Fitness: {results['best_fitness']:.4f}")
        
        # Mostrar evolución
        print(f"\n📊 Evolución por generación:")
        for gen_data in results['history']:
            print(f"  Gen {gen_data['generation']}: "
                  f"Promedio={gen_data['avg_fitness']:.3f}, "
                  f"Mejor={gen_data['best_fitness']:.3f}")
        
        # Mostrar parámetros
        params = results['parameters']
        print(f"\n⚙️ Parámetros usados:")
        print(f"  Población: {params['population_size']}")
        print(f"  Generaciones: {params['generations']}")
        print(f"  Mutación: {params['mutation_rate']}")
        print(f"  Modelo: {params['modelo_llm']}")
        print(f"  Temperatura: {params['temperatura']}")
        
        # Guardar resultados
        import json
        with open('covid_optimization_results.json', 'w') as f:
            # Convertir numpy types para serialización
            serializable_results = {}
            for key, value in results.items():
                if key == 'final_population':
                    continue  # Muy grande para guardar
                elif isinstance(value, float):
                    serializable_results[key] = float(value)
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados en 'covid_optimization_results.json'")
        
    except Exception as e:
        print(f"❌ Error ejecutando algoritmo: {e}")

def evaluar_prompt_especifico(optimizer):
    """
    Evaluar un prompt específico
    """
    print("\n🧪 Evaluación de prompt específico")
    print("=" * 40)
    
    try:
        # Generar prompt de ejemplo
        sample_texts = optimizer.sample_distinct(1)
        sample_prompts = optimizer.create_prompts(sample_texts)
        
        if not sample_prompts:
            print("❌ No se pudo generar prompt de ejemplo")
            return
        
        test_prompt = sample_prompts[0]
        
        print(f"📝 Evaluando prompt:")
        print(f"  Role: {test_prompt['role']}")
        print(f"  Task: {test_prompt['task_description']}")
        print(f"  Modifics: {test_prompt['modifics']}")
        print(f"  Texto: {test_prompt['texto'][:100]}...")
        
        # Evaluar fitness
        print(f"\n⏳ Enviando prompt al LLM...")
        fitness_scores = optimizer.evaluate_fitness([test_prompt])
        
        if fitness_scores:
            print(f"📊 Fitness obtenido: {fitness_scores[0]:.4f}")
            
            # Mostrar respuesta del LLM
            success, response = optimizer.ssh_executor.execute_prompt(
                test_prompt['full_prompt'],
                model=optimizer.modelo_llm,
                temperature=optimizer.temperatura
            )
            
            if success:
                print(f"\n🤖 Respuesta del LLM:")
                print(f"-" * 40)
                print(response)
                print(f"-" * 40)
            else:
                print(f"❌ Error obteniendo respuesta: {response}")
        else:
            print("❌ No se pudo evaluar el prompt")
            
    except Exception as e:
        print(f"❌ Error evaluando prompt: {e}")

def generar_datos_ejemplo(optimizer):
    """
    Generar datos de ejemplo para testing
    """
    print("\n📄 Generando datos de ejemplo...")
    
    try:
        # Usar datos de ejemplo del optimizador
        sample_data = optimizer._create_sample_covid_data()
        
        # Guardar en archivo
        sample_data.to_csv('sample_covid_data.csv', index=False)
        print(f"✅ Datos de ejemplo guardados en 'sample_covid_data.csv'")
        print(f"📊 Total de registros: {len(sample_data)}")
        
        # Mostrar algunos ejemplos
        print(f"\n📋 Ejemplos de textos:")
        for i, text in enumerate(sample_data['text'].head(3)):
            print(f"  {i+1}. {text}")
        
        # Generar prompts de ejemplo
        sample_texts = sample_data['text'].tolist()[:5]
        example_prompts = optimizer.create_prompts(sample_texts)
        
        # Guardar prompts
        import json
        with open('example_covid_prompts.json', 'w') as f:
            json.dump(example_prompts, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Prompts de ejemplo guardados en 'example_covid_prompts.json'")
        
    except Exception as e:
        print(f"❌ Error generando datos: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 