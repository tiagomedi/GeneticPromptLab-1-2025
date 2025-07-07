#!/usr/bin/env python3
"""
Prueba rápida del sistema GeneticPromptLab COVID actualizado
"""

import sys
import logging
from GeneticPromptLab.base_class import GeneticPromptLab

def test_initialization():
    """Prueba inicialización del sistema"""
    print("🧪 Probando inicialización del sistema...")
    
    try:
        optimizer = GeneticPromptLab(
            corpus_file='data/corpus.csv',
            population_size=3,
            generations=1,
            mutation_rate=0.5,
            sample_size=100,
            modelo_llm='llama3.1',
            temperatura=0.7,
            credentials_file='ssh_credentials.json'
        )
        
        print(f"✅ Sistema inicializado correctamente")
        print(f"   - Corpus: {len(optimizer.corpus_data)} textos")
        print(f"   - Roles: {len(optimizer.roles)} disponibles")
        print(f"   - Tasks: {len(optimizer.task_descriptions)} disponibles")
        print(f"   - Modifics: {len(optimizer.modifics_pool)} disponibles")
        
        return optimizer
        
    except Exception as e:
        print(f"❌ Error inicializando sistema: {e}")
        return None

def test_prompt_generation(optimizer):
    """Prueba generación de prompts"""
    print("\n🔄 Probando generación de prompts...")
    
    try:
        # Generar datos de muestra
        sample_texts = optimizer.sample_distinct(2)
        print(f"✅ Muestreados {len(sample_texts)} textos")
        
        # Crear prompts
        prompts = optimizer.create_prompts(sample_texts)
        print(f"✅ Generados {len(prompts)} prompts")
        
        # Mostrar ejemplo
        if prompts:
            example = prompts[0]
            print(f"\n📝 Ejemplo de prompt generado:")
            print(f"   Role: {example['role']}")
            print(f"   Task: {example['task_description']}")
            print(f"   Modifics: {example['modifics']}")
            print(f"   Texto: {example['texto'][:80]}...")
        
        return prompts
        
    except Exception as e:
        print(f"❌ Error generando prompts: {e}")
        return None

def test_ssh_connection(optimizer):
    """Prueba conexión SSH opcional"""
    print("\n🔗 Probando conexión SSH (opcional)...")
    
    try:
        # Intentar test básico
        ssh_status = optimizer.ssh_executor.test_setup()
        
        if ssh_status:
            print("✅ Conexión SSH funcionando")
            return True
        else:
            print("⚠️ Conexión SSH no disponible (esto es normal si no tienes servidor configurado)")
            return False
            
    except Exception as e:
        print(f"⚠️ SSH no disponible: {e}")
        print("💡 Esto es normal si no tienes servidor remoto configurado")
        return False

def test_fitness_calculation(optimizer):
    """Prueba cálculo de fitness sin SSH"""
    print("\n📊 Probando cálculo de fitness...")
    
    try:
        # Simular respuesta del LLM
        sample_response = "COVID-19 relevance score: 75. This text shows symptoms like fever and cough."
        sample_text = "I have fever and cough symptoms"
        
        # Calcular fitness
        fitness = optimizer._calculate_fitness(sample_response, sample_text)
        print(f"✅ Fitness calculado: {fitness:.3f}")
        
        # Probar con diferentes respuestas
        test_cases = [
            ("Score: 90/100 - Clear COVID symptoms", "high fitness"),
            ("No COVID indicators found", "low fitness"),
            ("Pandemic-related content with health concerns", "medium fitness")
        ]
        
        print("\n📈 Pruebas de fitness:")
        for response, expected in test_cases:
            fitness = optimizer._calculate_fitness(response, sample_text)
            print(f"   {expected}: {fitness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error calculando fitness: {e}")
        return False

def test_mutation(optimizer):
    """Prueba mutación de prompts"""
    print("\n🧬 Probando mutación de prompts...")
    
    try:
        # Crear prompt base
        sample_texts = optimizer.sample_distinct(1)
        prompts = optimizer.create_prompts(sample_texts)
        
        if not prompts:
            print("❌ No se pudieron crear prompts para mutación")
            return False
        
        original_prompt = prompts[0]
        
        # Aplicar mutación
        mutated_prompts = optimizer.mutate_prompts([original_prompt], mutation_rate=1.0)
        
        if mutated_prompts:
            mutated = mutated_prompts[0]
            
            print(f"✅ Mutación aplicada")
            print(f"   Original role: {original_prompt['role']}")
            print(f"   Mutado role: {mutated['role']}")
            print(f"   Original modifics: {original_prompt['modifics']}")
            print(f"   Mutado modifics: {mutated['modifics']}")
            
            return True
        else:
            print("❌ No se pudo aplicar mutación")
            return False
            
    except Exception as e:
        print(f"❌ Error en mutación: {e}")
        return False

def main():
    """Función principal de prueba"""
    print("🚀 Prueba del Sistema GeneticPromptLab COVID")
    print("=" * 50)
    
    # Configurar logging mínimo
    logging.basicConfig(level=logging.WARNING)
    
    # Contador de pruebas
    tests_passed = 0
    total_tests = 5
    
    # Prueba 1: Inicialización
    optimizer = test_initialization()
    if optimizer:
        tests_passed += 1
    else:
        print("❌ Fallo crítico en inicialización")
        return False
    
    # Prueba 2: Generación de prompts
    prompts = test_prompt_generation(optimizer)
    if prompts:
        tests_passed += 1
    
    # Prueba 3: Conexión SSH (opcional)
    ssh_working = test_ssh_connection(optimizer)
    if ssh_working:
        tests_passed += 1
    else:
        print("💡 SSH no disponible, continuando con pruebas locales...")
    
    # Prueba 4: Cálculo de fitness
    if test_fitness_calculation(optimizer):
        tests_passed += 1
    
    # Prueba 5: Mutación
    if test_mutation(optimizer):
        tests_passed += 1
    
    # Resultado final
    print(f"\n📊 Resultados de las pruebas:")
    print(f"   Pruebas pasadas: {tests_passed}/{total_tests}")
    print(f"   Porcentaje: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed >= 4:  # 4/5 es suficiente (SSH es opcional)
        print("\n🎉 ¡Sistema funcionando correctamente!")
        print("💡 Puedes usar example_covid_detection.py para ejemplo completo")
        return True
    else:
        print("\n❌ Sistema tiene problemas que necesitan ser resueltos")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 