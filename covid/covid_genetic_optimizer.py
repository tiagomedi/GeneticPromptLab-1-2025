#!/usr/bin/env python3
"""
Optimizador genético de prompts para COVID-19 usando Llama3
"""

import json
import random
import time
import pandas as pd
import numpy as np
import sys
from typing import List, Dict, Any, Tuple

class CovidGeneticOptimizer:
    def __init__(self, 
                 corpus_file: str,
                 population_size: int = 10,
                 generations: int = 5,
                 mutation_rate: float = 0.1,
                 sample_size: int = 1000,
                 modelo_llm: str = "llama3.1",
                 temperatura: float = 0.7):
        
        self.corpus_file = corpus_file
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.sample_size = min(sample_size, 1000)  # Limitar a 1000 para pruebas
        self.modelo_llm = modelo_llm
        self.temperatura = temperatura
        
        # Definir roles disponibles
        self.roles = [
            "researcher", "expert", "healthcare worker", "epidemiologist", 
            "public health official", "scientist", "medical professional", 
            "data analyst", "health journalist", "policy maker"
        ]
        
        # Definir descripciones de tareas
        self.task_descriptions = [
            "analyzing COVID-19 related data and trends",
            "generating informative content about pandemic impacts",
            "creating educational material about health measures",
            "developing reports on virus transmission patterns",
            "producing summaries of public health policies",
            "writing explanatory content about vaccination campaigns",
            "documenting healthcare system responses",
            "creating awareness content about prevention strategies"
        ]
        
        # Definir modificaciones que van a evolucionar genéticamente
        self.modifics_base = [
            "with focus on recent scientific findings",
            "emphasizing community-based approaches",
            "highlighting data-driven insights",
            "considering vulnerable populations",
            "from a global health perspective",
            "with attention to policy implications",
            "focusing on prevention and mitigation",
            "incorporating interdisciplinary viewpoints",
            "emphasizing evidence-based recommendations",
            "addressing public health challenges",
            "considering economic and social impacts",
            "with emphasis on healthcare system resilience",
            "highlighting innovation in medical responses",
            "focusing on long-term pandemic preparedness",
            "considering environmental health factors"
        ]
        
        # Cargar datos
        print("📊 Cargando corpus de datos...")
        self.data = pd.read_csv(corpus_file)
        
        # Detectar la columna de texto (la primera columna disponible)
        text_columns = [col for col in self.data.columns if self.data[col].dtype == 'object']
        if not text_columns:
            raise Exception("No se encontraron columnas de texto en el corpus")
        
        self.text_column = text_columns[0]  # Usar la primera columna de texto
        print(f"   Usando columna: '{self.text_column}'")
        
        # Filtrar datos válidos (no nulos)
        self.data = self.data.dropna(subset=[self.text_column])
        
        if len(self.data) > self.sample_size:
            self.data = self.data.sample(n=self.sample_size, random_state=42)
        print(f"   Usando {len(self.data)} muestras para optimización")
        
        # Configurar conexión SSH
        if not self._setup_ssh():
            raise Exception("No se pudo establecer conexión SSH")
    
    def _setup_ssh(self) -> bool:
        """Configurar conexión SSH"""
        try:
            from setup_ssh_tunnel_auto import RemoteSSHExecutor
            self.ssh = RemoteSSHExecutor()
            
            if not self.ssh.connect():
                print("❌ Error conectando al servidor")
                return False
            
            # Obtener ruta de ollama
            success, output = self.ssh.run_command("which ollama")
            if not success or "ollama" not in output:
                print("❌ Ollama no está instalado")
                return False
                
            self.ollama_path = output.strip()
            print(f"✅ Ollama encontrado en: {self.ollama_path}")
            
            # Verificar Llama3
            if not self.ssh.test_llama():
                print("❌ Error verificando Llama3")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error configurando SSH: {e}")
            return False
    
    def _generate_initial_population(self) -> List[Dict[str, Any]]:
        """Generar población inicial de prompts con estructura role + task description + modifics"""
        print("   📝 Generando prompts con estructura role + task description + modifics...")
        
        population = []
        
        # Seleccionar textos de referencia para inspirar variaciones
        reference_texts = self.data[self.text_column].sample(n=min(self.population_size, 20), random_state=42).tolist()
        
        for i in range(self.population_size):
            # Seleccionar componentes para la estructura
            role = random.choice(self.roles)
            task_description = random.choice(self.task_descriptions)
            
            # Crear modificaciones evolucionables (1-3 modificaciones por prompt)
            num_modifics = random.randint(1, 3)
            selected_modifics = random.sample(self.modifics_base, num_modifics)
            modifics = ", ".join(selected_modifics)
            
            # Obtener texto de referencia
            texto = reference_texts[i % len(reference_texts)]
            
            # Crear prompt estructurado
            prompt_structure = {
                "role": role,
                "task_description": task_description,
                "modifics": modifics,
                "texto": texto[:200]  # Limitar longitud del texto
            }
            
            population.append(prompt_structure)
            
            print(f"   ✅ Prompt {i+1}:")
            print(f"      Role: {role}")
            print(f"      Task: {task_description}")
            print(f"      Modifics: {modifics}")
            print()
        
        return population
    
    def _create_llama_payload(self, prompt_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Crear el payload para Llama3 con la estructura role + task description + modifics"""
        role = prompt_structure["role"]
        task_description = prompt_structure["task_description"]
        modifics = prompt_structure["modifics"]
        texto = prompt_structure["texto"]
        
        # Construir el prompt final estructurado
        final_prompt = f"Role: {role}\nTask Description: {task_description}\nModifications: {modifics}"
        
        payload = {
            "model": self.modelo_llm,
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
            "temperature": self.temperatura
        }
        
        return payload
    
    def _evaluate_prompt(self, prompt_structure: Dict[str, Any], reference_texts: List[str]) -> float:
        """Evaluar un prompt estructurado midiendo qué tan bien genera textos similares al corpus"""
        similarity_scores = []
        
        role = prompt_structure["role"]
        task = prompt_structure["task_description"]
        modifics = prompt_structure["modifics"]
        print(f"      🔍 Evaluando prompt:")
        print(f"         Role: {role}")
        print(f"         Task: {task[:50]}...")
        print(f"         Modifics: {modifics[:50]}...")
        
        for i, reference_text in enumerate(reference_texts[:3]):  # Limitar a 3 textos de referencia
            print(f"      📝 Generando texto {i+1}/3...")
            
            # Crear payload estructurado
            payload = self._create_llama_payload(prompt_structure)
            
            # Generar texto usando el prompt estructurado
            success, generated_text = self.ssh.run_ollama_structured_command(payload)
            
            if not success:
                print(f"      ❌ Error: {generated_text}")
                continue
            
            print(f"      ✅ Texto generado: {generated_text[:100]}...")
            
            # Calcular similitud basada en palabras clave comunes
            similarity = self._calculate_similarity(reference_text, generated_text)
            similarity_scores.append(similarity)
            
            print(f"      📊 Similitud: {similarity:.3f}")
        
        # Retornar similitud promedio
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            print(f"      📈 Similitud promedio: {avg_similarity:.3f}")
            return avg_similarity
        else:
            print(f"      ⚠️ No se pudo calcular similitud")
            return 0.0
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud simple entre dos textos"""
        # Convertir a minúsculas y dividir en palabras
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calcular intersección y unión
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Similitud de Jaccard
        if len(union) == 0:
            return 0.0
        
        jaccard_similarity = len(intersection) / len(union)
        
        # También considerar longitud similar (penalizar diferencias muy grandes)
        length_similarity = 1.0 - abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1)
        
        # Combinar ambas métricas
        combined_similarity = (jaccard_similarity * 0.7) + (length_similarity * 0.3)
        
        return combined_similarity
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realizar crossover entre dos prompts estructurados, modificando principalmente los modifics"""
        # Role y task description pueden intercambiarse, pero modifics es lo que más evoluciona
        child = {
            "role": random.choice([parent1["role"], parent2["role"]]),
            "task_description": random.choice([parent1["task_description"], parent2["task_description"]]),
            "texto": random.choice([parent1["texto"], parent2["texto"]])
        }
        
        # Combinar modifics de ambos padres
        modifics1 = parent1["modifics"].split(", ")
        modifics2 = parent2["modifics"].split(", ")
        
        # Crear nueva combinación de modifics
        all_modifics = modifics1 + modifics2
        # Eliminar duplicados manteniendo orden
        unique_modifics = list(dict.fromkeys(all_modifics))
        
        # Seleccionar 1-3 modifics para el hijo
        num_modifics = random.randint(1, min(3, len(unique_modifics)))
        selected_modifics = random.sample(unique_modifics, num_modifics)
        child["modifics"] = ", ".join(selected_modifics)
        
        return child
    
    def _mutate(self, prompt_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Mutar un prompt estructurado, enfocándose principalmente en modifics"""
        if random.random() > self.mutation_rate:
            return prompt_structure
        
        # Crear una copia del prompt
        mutated = prompt_structure.copy()
        
        # Mutar role ocasionalmente (20% de probabilidad)
        if random.random() < 0.2:
            mutated["role"] = random.choice(self.roles)
        
        # Mutar task description ocasionalmente (30% de probabilidad)
        if random.random() < 0.3:
            mutated["task_description"] = random.choice(self.task_descriptions)
        
        # Mutar modifics frecuentemente (80% de probabilidad) - aquí está la evolución principal
        if random.random() < 0.8:
            current_modifics = mutated["modifics"].split(", ")
            
            # Estrategias de mutación para modifics
            mutation_strategy = random.choice(["replace", "add", "remove", "shuffle"])
            
            if mutation_strategy == "replace" and current_modifics:
                # Reemplazar una modificación existente
                idx = random.randint(0, len(current_modifics) - 1)
                new_modific = random.choice(self.modifics_base)
                current_modifics[idx] = new_modific
                
            elif mutation_strategy == "add" and len(current_modifics) < 3:
                # Agregar una nueva modificación
                new_modific = random.choice(self.modifics_base)
                if new_modific not in current_modifics:
                    current_modifics.append(new_modific)
                    
            elif mutation_strategy == "remove" and len(current_modifics) > 1:
                # Remover una modificación
                idx = random.randint(0, len(current_modifics) - 1)
                current_modifics.pop(idx)
                
            elif mutation_strategy == "shuffle" and len(current_modifics) > 1:
                # Reorganizar modificaciones
                random.shuffle(current_modifics)
            
            mutated["modifics"] = ", ".join(current_modifics)
        
        return mutated
    
    def _format_final_prompt(self, prompt_structure: Dict[str, Any]) -> str:
        """Formatear el prompt final con la estructura requerida"""
        role = prompt_structure["role"]
        task_description = prompt_structure["task_description"]
        modifics = prompt_structure["modifics"]
        
        final_prompt = f"Role: {role}\nTask Description: {task_description}\nModifications: {modifics}"
        return final_prompt
    
    def optimize(self) -> Dict[str, Any]:
        """Ejecutar optimización genética"""
        print("\n🧬 Iniciando optimización genética...")
        
        # Población inicial
        print("👥 Generando población inicial...")
        population = self._generate_initial_population()
        
        best_prompt = None
        best_fitness = 0
        history = []
        
        # Evolución
        for generation in range(self.generations):
            print(f"\n🔄 Generación {generation + 1}/{self.generations}")
            
            # Evaluar población
            fitness_scores = []
            for i, prompt_structure in enumerate(population):
                print(f"   Evaluando prompt {i + 1}/{len(population)}...")
                fitness = self._evaluate_prompt(prompt_structure, self.data[self.text_column].tolist())
                fitness_scores.append(fitness)
                print(f"   Fitness: {fitness:.2f}")
                
                # Actualizar mejor prompt
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_prompt = prompt_structure
            
            # Guardar historia
            history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': sum(fitness_scores) / len(fitness_scores)
            })
            
            # Selección
            parents = []
            for _ in range(self.population_size):
                # Usar un torneo más pequeño si la población es pequeña
                tournament_size = min(3, len(population))
                tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
                winner = max(tournament, key=lambda x: x[1])[0]
                parents.append(winner)
            
            # Nueva generación
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
            
            print(f"   Mejor fitness actual: {best_fitness:.2f}")
            if best_prompt:
                print(f"   Mejor prompt estructura:")
                print(f"     Role: {best_prompt['role']}")
                print(f"     Task: {best_prompt['task_description']}")
                print(f"     Modifics: {best_prompt['modifics']}")
        
        # Formatear prompt final
        final_prompt_text = self._format_final_prompt(best_prompt) if best_prompt else None
        
        # Guardar resultados
        results = {
            'best_prompt_structure': best_prompt,
            'best_prompt_formatted': final_prompt_text,
            'best_fitness': best_fitness,
            'history': history,
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'sample_size': self.sample_size,
                'modelo_llm': self.modelo_llm,
                'temperatura': self.temperatura
            }
        }
        
        with open('covid_prompts.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Optimización completada")
        print(f"   Mejor prompt estructurado:")
        if best_prompt:
            print(f"     Role: {best_prompt['role']}")
            print(f"     Task: {best_prompt['task_description']}")
            print(f"     Modifics: {best_prompt['modifics']}")
        print(f"   Fitness: {best_fitness:.2f}")
        
        print(f"\n📝 Prompt final formateado:")
        print(final_prompt_text)
        
        # Cerrar conexión SSH
        self.ssh.close()
        
        return results

def main():
    """Función principal"""
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimizador genético de prompts COVID-19')
    parser.add_argument('--corpus', type=str, default='corpus.csv',
                      help='Archivo del corpus (default: corpus.csv)')
    parser.add_argument('--population', type=int, default=5,
                      help='Tamaño de población (default: 5)')
    parser.add_argument('--generations', type=int, default=3,
                      help='Número de generaciones (default: 3)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                      help='Tasa de mutación (default: 0.1)')
    parser.add_argument('--sample-size', type=int, default=500,
                      help='Tamaño de muestra del corpus (default: 500)')
    parser.add_argument('--modelo-llm', type=str, default='llama3.1',
                      help='Modelo LLM a usar (default: llama3.1)')
    parser.add_argument('--temperatura', type=float, default=0.7,
                      help='Temperatura para la generación (default: 0.7)')
    
    args = parser.parse_args()
    
    try:
        optimizer = CovidGeneticOptimizer(
            corpus_file=args.corpus,
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            sample_size=args.sample_size,
            modelo_llm=args.modelo_llm,
            temperatura=args.temperatura
        )
        
        optimizer.optimize()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 