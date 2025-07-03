#!/usr/bin/env python3
"""
Optimizador genético para análisis de COVID-19 usando Mistral vía SSH
"""

import requests
import json
import random
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, cast, Any
import time
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Agregar la ruta del proyecto principal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GeneticPromptLab.base_class import GeneticPromptLab

class CovidGeneticOptimizer(GeneticPromptLab):
    """
    Optimizador genético especializado para análisis de COVID-19
    """
    
    def __init__(self, 
                 corpus_file: str = "corpus.csv",
                 ssh_tunnel_port: int = 11435,
                 model_name: str = "mistral",
                 sample_size: int = 1000,
                 init_population_size: int = 8,
                 generations: int = 10):
        
        self.corpus_file = corpus_file
        self.ssh_tunnel_port = ssh_tunnel_port
        self.model_name = model_name
        self.sample_size = sample_size
        self.init_population_size = init_population_size
        self.generations = generations
        
        # URL del LLM a través del túnel SSH
        self.llm_api_url = f"http://localhost:{ssh_tunnel_port}/api/chat"
        
        # Roles específicos para COVID-19
        self.roles = [
            "a healthcare worker",
            "a public health expert", 
            "a epidemiologist",
            "a government official",
            "a researcher",
            "a citizen affected by COVID",
            "a policy maker",
            "a medical professional",
            "a data scientist",
            "a social scientist"
        ]
        
        # Cargar y procesar datos
        self.corpus_data = self._load_corpus()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.text_embeddings = None
        self.sampled_texts: Optional[List[str]] = None
        
        print(f"✅ CovidGeneticOptimizer inicializado")
        print(f"   Corpus: {len(self.corpus_data)} documentos")
        print(f"   Modelo: {model_name}")
        print(f"   Puerto túnel SSH: {ssh_tunnel_port}")
    
    def _load_corpus(self) -> pd.DataFrame:
        """
        Carga el archivo corpus.csv
        """
        try:
            print(f"📁 Cargando corpus desde: {self.corpus_file}")
            
            # Leer CSV - asumiendo que tiene al menos una columna de texto
            df = pd.read_csv(self.corpus_file)
            
            # Detectar columna de texto (la más larga en promedio)
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_length = df[col].astype(str).str.len().mean()
                    text_columns.append((col, avg_length))
            
            if not text_columns:
                raise ValueError("No se encontraron columnas de texto en el corpus")
            
            # Seleccionar la columna con texto más largo
            main_text_column = max(text_columns, key=lambda x: x[1])[0]
            
            print(f"   Columna de texto principal: {main_text_column}")
            print(f"   Tamaño del corpus: {len(df)} documentos")
            
            # Limpiar datos
            df = df.dropna(subset=[main_text_column])
            df['text'] = df[main_text_column].astype(str)
            
            # Filtrar textos muy cortos o muy largos
            text_series = cast(pd.Series, df['text'])
            df = df[text_series.str.len() > 50]
            df = df[text_series.str.len() < 5000]
            
            print(f"   Documentos después de limpieza: {len(df)}")
            
            # Retornar DataFrame explícitamente
            result_df = pd.DataFrame({'text': df['text'].tolist()})
            return result_df
            
        except Exception as e:
            print(f"❌ Error al cargar corpus: {e}")
            # Crear datos dummy si falla
            return pd.DataFrame({
                'text': [
                    "COVID-19 pandemic has affected global health systems significantly.",
                    "Vaccination campaigns have been crucial in controlling the spread.",
                    "Economic impacts of lockdowns have been substantial worldwide.",
                    "Mental health effects during pandemic require attention.",
                    "Healthcare workers faced unprecedented challenges during COVID-19."
                ]
            })
    
    def _prepare_text_embeddings(self):
        """
        Prepara embeddings para clustering y muestreo diverso
        """
        if self.text_embeddings is not None:
            return
            
        print("🔄 Preparando embeddings de texto...")
        
        # Usar TF-IDF para embeddings simples
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        
        # Tomar muestra para embeddings si el corpus es muy grande
        if len(self.corpus_data) > self.sample_size:
            sample_indices = np.random.choice(
                len(self.corpus_data), 
                self.sample_size, 
                replace=False
            )
            sample_texts = self.corpus_data.iloc[sample_indices]['text'].tolist()
        else:
            sample_texts = self.corpus_data['text'].tolist()
        
        self.text_embeddings = self.vectorizer.fit_transform(sample_texts)
        self.sampled_texts = sample_texts
        
        print(f"✅ Embeddings preparados: {self.text_embeddings.shape}")  # type: ignore
    
    def _connect_to_mistral(self, messages: List[Dict[str, str]], temperature: float = 0.8) -> Optional[str]:
        """
        Conecta a Mistral vía túnel SSH
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.llm_api_url, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")
                return content
            else:
                print(f"❌ Error en API: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de conexión: {e}")
            return None
    
    def sample_distinct(self, n: int) -> List[str]:
        """
        Muestrea n textos diversos usando clustering
        """
        self._prepare_text_embeddings()
        
        # Verificar que sampled_texts no sea None
        if self.sampled_texts is None:
            return []
        
        if len(self.sampled_texts) <= n:
            return self.sampled_texts
        
        # Verificar que text_embeddings no sea None
        if self.text_embeddings is None:
            return self.sampled_texts[:n]
        
        # Clustering para diversidad
        kmeans = KMeans(n_clusters=min(n, len(self.sampled_texts)), random_state=42)
        cluster_labels = kmeans.fit_predict(self.text_embeddings)
        
        # Seleccionar un texto por cluster
        selected_texts = []
        for cluster_id in range(n):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Seleccionar el texto más cercano al centroide del cluster
                cluster_embeddings = self.text_embeddings[cluster_indices]
                centroid = cluster_embeddings.mean(axis=0)  # type: ignore
                
                # Convertir matrices sparse a arrays numpy densos
                from scipy.sparse import issparse
                
                if issparse(cluster_embeddings):
                    cluster_embeddings_array = cluster_embeddings.toarray()  # type: ignore
                else:
                    cluster_embeddings_array = np.asarray(cluster_embeddings)
                
                if issparse(centroid):
                    centroid_array = centroid.toarray().reshape(1, -1)  # type: ignore
                else:
                    centroid_array = np.asarray(centroid).reshape(1, -1)
                
                similarities = cosine_similarity(cluster_embeddings_array, centroid_array)
                best_idx = cluster_indices[similarities.argmax()]
                selected_texts.append(self.sampled_texts[best_idx])
        
        return selected_texts[:n]
    
    def create_prompts(self, texts: List[str]) -> List[str]:
        """
        Genera prompts usando Mistral para cada texto
        """
        prompts = []
        
        for i, text in enumerate(texts):
            role = random.choice(self.roles)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a prompt engineering expert specializing in COVID-19 analysis. "
                        "Generate clear, structured prompts for AI systems to analyze COVID-related content. "
                        "Each prompt should:\n"
                        "1. Define a specific role/perspective\n"
                        "2. Specify the analytical task\n"
                        "3. Focus on COVID-19 related aspects\n"
                        "4. Be concise but comprehensive\n"
                        "Return only the prompt text, no additional explanation."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate a COVID-19 analysis prompt for the role '{role}' based on this text:\n\n"
                        f"Text: \"{text[:500]}...\"\n\n"
                        f"Focus on extracting insights about pandemic impacts, health measures, "
                        f"or social/economic effects."
                    )
                }
            ]
            
            print(f"🔄 Generando prompt {i+1}/{len(texts)} para rol: {role}")
            
            response = self._connect_to_mistral(messages, temperature=0.8)
            
            if response:
                # Limpiar la respuesta
                prompt = self._clean_prompt(response)
                prompts.append(prompt)
                print(f"✅ Prompt generado: {prompt[:100]}...")
            else:
                # Prompt de fallback
                fallback_prompt = f"As {role}, analyze the following COVID-19 related content and provide insights about pandemic impacts, health measures, and social effects."
                prompts.append(fallback_prompt)
                print(f"⚠️ Usando prompt de fallback")
            
            time.sleep(1)  # Evitar sobrecarga del servidor
        
        return prompts
    
    def _clean_prompt(self, prompt: str) -> str:
        """
        Limpia y formatea el prompt generado
        """
        # Remover comillas si están presentes
        cleaned = re.sub(r'^["\'](.*)["\']$', r'\1', prompt.strip())
        
        # Remover prefijos comunes
        prefixes_to_remove = [
            "Here is a prompt:",
            "Here's a prompt:",
            "Prompt:",
            "Generated prompt:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        return cleaned
    
    def generate_init_prompts(self, n: Optional[int] = None) -> List[str]:
        """
        Genera población inicial de prompts
        """
        if n is None:
            n = self.init_population_size
        
        print(f"🚀 Generando {n} prompts iniciales...")
        
        # Muestrear textos diversos
        selected_texts = self.sample_distinct(n)
        
        # Generar prompts
        prompts = self.create_prompts(selected_texts)
        
        print(f"✅ {len(prompts)} prompts generados exitosamente")
        return prompts
    
    def genetic_algorithm(self, mutation_rate: float = 0.1) -> List[str]:
        """
        Ejecuta el algoritmo genético (implementación básica)
        """
        print(f"🧬 Iniciando algoritmo genético para {self.generations} generaciones")
        
        # Generar población inicial
        population = self.generate_init_prompts()
        
        # Por ahora, solo retorna la población inicial
        # TODO: Implementar fitness, selección, cruzamiento y mutación
        return population
    
    def evaluate_fitness(self, prompts: List[str]) -> List[float]:
        """
        Evalúa el fitness de los prompts (placeholder)
        """
        # TODO: Implementar evaluación de fitness real
        return [random.random() for _ in prompts]
    
    def select_top_prompts(self, fitness_scores: List[float], population: List[str], 
                          prompt_answers_list: Optional[List[Any]] = None, top_fraction: float = 0.5) -> List[str]:
        """
        Selecciona los mejores prompts (placeholder)
        """
        # TODO: Implementar selección real
        n_select = max(1, int(len(population) * top_fraction))
        return population[:n_select]
    
    def crossover_using_gpt(self, prompts: List[str], questions_list: List[str], 
                           correct_answers_list: List[str], top_prompts_answers_list: Optional[List[Any]] = None) -> List[str]:
        """
        Realiza cruzamiento usando GPT (placeholder)
        """
        # TODO: Implementar cruzamiento real
        return prompts
    
    def mutate_prompts(self, prompts: List[str], mutation_rate: float = 0.1) -> List[str]:
        """
        Muta prompts (placeholder)
        """
        # TODO: Implementar mutación real
        return prompts
    
    def save_results(self, prompts: List[str], filename: str = "covid_prompts.json"):
        """
        Guarda los resultados en un archivo JSON
        """
        results = [
            {
                "prompt": prompt,
                "role": "generated",
                "fitness": 0.0,
                "generation": 0
            }
            for prompt in prompts
        ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados guardados en: {filename}")

def main():
    """
    Función principal para probar el optimizador
    """
    print("🦠 COVID-19 Genetic Prompt Optimizer")
    print("=" * 50)
    
    # Crear optimizador
    optimizer = CovidGeneticOptimizer(
        corpus_file="corpus.csv",
        ssh_tunnel_port=11435,
        model_name="mistral",
        sample_size=500,
        init_population_size=5,
        generations=3
    )
    
    try:
        # Ejecutar algoritmo genético
        final_prompts = optimizer.genetic_algorithm()
        
        # Guardar resultados
        optimizer.save_results(final_prompts)
        
        print("\n✅ Optimización completada!")
        print(f"   Prompts finales: {len(final_prompts)}")
        
    except Exception as e:
        print(f"❌ Error durante la optimización: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 