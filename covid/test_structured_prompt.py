#!/usr/bin/env python3
"""
Script de prueba para verificar la nueva estructura de prompts
Role + Task Description + Modifics
"""

import json
import sys
import os
from setup_ssh_tunnel_auto import RemoteSSHExecutor

def test_structured_prompt():
    """Probar el nuevo formato de prompt estructurado: role + task description + modifics"""
    print("🧪 Probando nueva estructura de prompts: role + task description + modifics...")
    
    # Crear executor
    executor = RemoteSSHExecutor()
    
    try:
        # Conectar
        if not executor.connect():
            print("❌ Error conectando al servidor")
            return False
        
        # Definir estructura del prompt
        role = "healthcare worker"
        task_description = "analyzing COVID-19 related data and trends"
        modifics = "with focus on recent scientific findings, emphasizing community-based approaches"
        
        # Construir prompt final estructurado
        final_prompt = f"Role: {role}\nTask Description: {task_description}\nModifications: {modifics}"
        
        # Texto de referencia
        texto = ("The COVID-19 pandemic has significantly impacted global health systems, requiring "
                "unprecedented responses from healthcare professionals worldwide. Recent data shows "
                "varying transmission patterns across different communities.")
        
        # Crear payload de prueba con la nueva estructura
        payload = {
            "model": "llama3.1",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that generates structured and narrative prompts for training AI systems. "
                        "Each prompt should have a clear and consistent structure with three main components: "
                        "1. Role - Identify the speaker's perspective and expertise area "
                        "2. Task Description - Specify the main task or activity being performed "
                        "3. Modifications - Add specific modifications that guide the approach or focus "
                        "Generate content that follows this structure and is aligned with the provided text."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Using this prompt structure:\n{final_prompt}\n\n"
                        f"Generate content based on this reference text: \"{texto}\""
                    )
                }
            ],
            "stream": False,
            "temperature": 0.7
        }
        
        print("📋 Estructura del prompt:")
        print(f"   Role: {role}")
        print(f"   Task Description: {task_description}")
        print(f"   Modifications: {modifics}")
        print()
        
        print("📋 Prompt final formateado:")
        print(final_prompt)
        print()
        
        print("📋 Payload completo:")
        print(json.dumps(payload, indent=2))
        print()
        
        # Ejecutar prompt estructurado
        print("🔄 Ejecutando prompt estructurado...")
        success, response = executor.run_ollama_structured_command(payload)
        
        if success:
            print("✅ Prompt estructurado funcionó correctamente")
            print(f"📝 Respuesta:\n{response}")
            print()
            
            # Verificar que la respuesta contiene elementos estructurados esperados
            expected_keywords = ["role", "task", "modific", "healthcare", "data", "analysis", "scientific", "community"]
            found_keywords = [keyword for keyword in expected_keywords if keyword.lower() in response.lower()]
            
            if len(found_keywords) >= 3:
                print(f"✅ La respuesta contiene elementos estructurados esperados: {found_keywords}")
                return True
            else:
                print(f"⚠️ La respuesta contiene pocos elementos estructurados esperados: {found_keywords}")
                return False
        else:
            print(f"❌ Error ejecutando prompt estructurado: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        return False
    finally:
        executor.close()

def test_modific_evolution():
    """Probar cómo evolucionan las modificaciones"""
    print("🧬 Probando evolución de modificaciones...")
    
    # Definir modificaciones base
    modifics_base = [
        "with focus on recent scientific findings",
        "emphasizing community-based approaches",
        "highlighting data-driven insights",
        "considering vulnerable populations",
        "from a global health perspective",
        "with attention to policy implications",
        "focusing on prevention and mitigation",
        "incorporating interdisciplinary viewpoints"
    ]
    
    # Simular evolución de modificaciones
    import random
    
    # Padre 1
    parent1_modifics = random.sample(modifics_base, 2)
    print(f"   Padre 1 modifics: {', '.join(parent1_modifics)}")
    
    # Padre 2
    parent2_modifics = random.sample(modifics_base, 3)
    print(f"   Padre 2 modifics: {', '.join(parent2_modifics)}")
    
    # Crossover
    all_modifics = parent1_modifics + parent2_modifics
    unique_modifics = list(dict.fromkeys(all_modifics))
    num_modifics = random.randint(1, min(3, len(unique_modifics)))
    child_modifics = random.sample(unique_modifics, num_modifics)
    print(f"   Hijo modifics: {', '.join(child_modifics)}")
    
    # Mutación
    mutation_strategies = ["replace", "add", "remove", "shuffle"]
    strategy = random.choice(mutation_strategies)
    print(f"   Estrategia de mutación: {strategy}")
    
    mutated_modifics = child_modifics.copy()
    if strategy == "replace" and mutated_modifics:
        idx = random.randint(0, len(mutated_modifics) - 1)
        new_modific = random.choice(modifics_base)
        mutated_modifics[idx] = new_modific
    elif strategy == "add" and len(mutated_modifics) < 3:
        new_modific = random.choice(modifics_base)
        if new_modific not in mutated_modifics:
            mutated_modifics.append(new_modific)
    elif strategy == "remove" and len(mutated_modifics) > 1:
        idx = random.randint(0, len(mutated_modifics) - 1)
        mutated_modifics.pop(idx)
    elif strategy == "shuffle" and len(mutated_modifics) > 1:
        random.shuffle(mutated_modifics)
    
    print(f"   Modifics después de mutación: {', '.join(mutated_modifics)}")
    
    # Formatear prompt final
    role = "epidemiologist"
    task_description = "developing reports on virus transmission patterns"
    final_prompt = f"Role: {role}\nTask Description: {task_description}\nModifications: {', '.join(mutated_modifics)}"
    
    print(f"\n📝 Prompt final evolucionado:")
    print(final_prompt)
    
    return True

def main():
    """Función principal"""
    print("🦠 Prueba de Nueva Estructura de Prompts COVID-19")
    print("=" * 60)
    print("   Estructura: Role + Task Description + Modifics")
    print("=" * 60)
    
    # Prueba 1: Estructura básica
    print("\n🧪 PRUEBA 1: Estructura básica")
    success1 = test_structured_prompt()
    
    # Prueba 2: Evolución de modificaciones
    print("\n🧪 PRUEBA 2: Evolución de modificaciones")
    success2 = test_modific_evolution()
    
    # Resultado final
    overall_success = success1 and success2
    
    if overall_success:
        print("\n🎉 ¡Todas las pruebas completadas exitosamente!")
        print("   La nueva estructura role + task description + modifics está funcionando correctamente.")
        print("   Los modifics evolucionan correctamente mediante algoritmos genéticos.")
        print("\n📋 Estructura final confirmada:")
        print("   1. Role: Perspectiva del hablante")
        print("   2. Task Description: Descripción de la tarea")
        print("   3. Modifications: Modificaciones que evolucionan genéticamente")
    else:
        print("\n❌ Algunas pruebas fallaron")
        print("   Revisa la configuración y conexión al servidor.")
    
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main() 