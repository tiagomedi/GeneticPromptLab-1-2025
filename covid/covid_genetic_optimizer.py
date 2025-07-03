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
                 sample_size: int = 1000):
        
        self.corpus_file = corpus_file
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.sample_size = min(sample_size, 1000)  # Limitar a 1000 para pruebas
        
        # Cargar datos
        print("📊 Cargando corpus de datos...")
        self.data = pd.read_csv(corpus_file)
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
    
    def _generate_initial_population(self) -> List[str]:
        """Generar población inicial de prompts"""
        base_prompts = [
            "Analiza el siguiente texto y determina si menciona COVID-19 o coronavirus: ",
            "¿Este texto contiene información sobre COVID-19? Responde 'Sí' o 'No': ",
            "Identifica si el siguiente texto está relacionado con COVID-19: ",
            "Lee este texto y determina si trata sobre coronavirus: ",
            "Evalúa si este texto menciona la pandemia de COVID-19: "
        ]
        
        population = []
        for _ in range(self.population_size):
            if random.random() < 0.7 and base_prompts:  # 70% chance de usar un prompt base
                prompt = random.choice(base_prompts)
            else:  # 30% chance de generar uno nuevo
                success, new_prompt = self.ssh.run_ollama_command(
                    "Genera un prompt corto para detectar si un texto menciona COVID-19"
                )
                if success and new_prompt.strip():
                    prompt = new_prompt.strip()
                else:
                    prompt = random.choice(base_prompts)
            
            population.append(prompt)
        
        return population
    
    def _evaluate_prompt(self, prompt: str, texts: List[str]) -> float:
        """Evaluar un prompt usando Llama3"""
        correct = 0
        total = len(texts)
        
        for text in texts:
            # Enviar prompt y texto a Llama3
            full_prompt = f"{prompt} {text}"
            success, response = self.ssh.run_ollama_command(full_prompt)
            
            if not success:
                print(f"⚠️ Error evaluando texto: {text[:50]}...")
                continue
            
            # Analizar respuesta
            response = response.lower()
            contains_covid = any(term in text.lower() for term in ['covid', 'coronavirus', 'sars-cov-2'])
            
            # Verificar si la respuesta es correcta
            if contains_covid:
                if any(term in response for term in ['sí', 'si', 'yes', 'true', 'correcto']):
                    correct += 1
            else:
                if any(term in response for term in ['no', 'false', 'incorrecto']):
                    correct += 1
        
        return correct / total if total > 0 else 0
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Realizar crossover entre dos prompts"""
        # Dividir prompts en palabras
        words1 = parent1.split()
        words2 = parent2.split()
        
        # Punto de cruce aleatorio
        point = random.randint(1, min(len(words1), len(words2)))
        
        # Crear nuevo prompt
        new_words = words1[:point] + words2[point:]
        return ' '.join(new_words)
    
    def _mutate(self, prompt: str) -> str:
        """Mutar un prompt"""
        if random.random() > self.mutation_rate:
            return prompt
            
        # Solicitar a Llama3 que modifique el prompt
        mutation_prompt = f"Modifica ligeramente este prompt manteniendo su objetivo: {prompt}"
        success, new_prompt = self.ssh.run_ollama_command(mutation_prompt)
        
        if success and new_prompt.strip():
            return new_prompt.strip()
        return prompt
    
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
            for i, prompt in enumerate(population):
                print(f"   Evaluando prompt {i + 1}/{len(population)}...")
                fitness = self._evaluate_prompt(prompt, self.data['text'].tolist())
                fitness_scores.append(fitness)
                print(f"   Fitness: {fitness:.2f}")
                
                # Actualizar mejor prompt
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_prompt = prompt
            
            # Guardar historia
            history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': sum(fitness_scores) / len(fitness_scores)
            })
            
            # Selección
            parents = []
            for _ in range(self.population_size):
                tournament = random.sample(list(zip(population, fitness_scores)), 3)
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
            print(f"   Mejor prompt: {best_prompt}")
        
        # Guardar resultados
        results = {
            'best_prompt': best_prompt,
            'best_fitness': best_fitness,
            'history': history,
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'sample_size': self.sample_size
            }
        }
        
        with open('covid_prompts.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Optimización completada")
        print(f"   Mejor prompt: {best_prompt}")
        print(f"   Fitness: {best_fitness:.2f}")
        
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
    
    args = parser.parse_args()
    
    try:
        optimizer = CovidGeneticOptimizer(
            corpus_file=args.corpus,
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            sample_size=args.sample_size
        )
        
        optimizer.optimize()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 