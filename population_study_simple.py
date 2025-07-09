#!/usr/bin/env python3
"""
Script simplificado para estudiar métodos de generación de poblaciones iniciales
para optimización genética de prompts.

Enfoque: Comparar diversidad y características de diferentes métodos
"""

import pandas as pd
import numpy as np
import json
import random
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

class SimplePopulationStudy:
    """
    Estudio simplificado de métodos de generación de poblaciones iniciales
    """
    
    def __init__(self, corpus_file: str = 'data/corpus.csv', sample_size: int = 100):
        self.corpus_file = corpus_file
        self.sample_size = sample_size
        
        # Cargar datos
        self.corpus_data = self._load_corpus()
        
        # Generar embeddings simples (longitud de texto + características básicas)
        self.simple_embeddings = self._generate_simple_embeddings()
        
        # Componentes para prompts estructurados
        self.roles = [
            "epidemiologist", "medical researcher", "public health expert",
            "healthcare analyst", "disease surveillance specialist"
        ]
        
        self.task_descriptions = [
            "analyzing text content for COVID-19 related information",
            "identifying pandemic-related themes and patterns",
            "detecting health emergency indicators in text",
            "classifying content based on COVID-19 relevance"
        ]
        
        self.modifics_pool = [
            "with focus on symptom identification",
            "emphasizing transmission patterns",
            "highlighting prevention measures",
            "considering healthcare system impact",
            "from a public health surveillance perspective"
        ]
        
        print(f"✅ SimplePopulationStudy inicializado - Corpus: {len(self.corpus_data)} textos")
    
    def _load_corpus(self) -> pd.DataFrame:
        """Cargar corpus de datos"""
        try:
            df = pd.read_csv(self.corpus_file)
            print(f"✅ Corpus cargado: {len(df)} registros")
            
            # Identificar columna de texto
            text_columns = [col for col in df.columns if 'text' in col.lower()]
            if text_columns:
                df = df.rename(columns={text_columns[0]: 'text'})
            elif 'text' not in df.columns:
                first_col = df.columns[0]
                df = df.rename(columns={first_col: 'text'})
            
            # Filtrar y limitar
            df = df.dropna(subset=['text'])
            df = df[df['text'].astype(str).str.len() > 10]
            
            if len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error cargando corpus: {e}")
            return self._create_sample_covid_data()
    
    def _create_sample_covid_data(self) -> pd.DataFrame:
        """Crear datos de ejemplo para COVID"""
        sample_texts = [
            "I have been experiencing fever, cough, and difficulty breathing.",
            "The COVID-19 pandemic has impacted global healthcare systems.",
            "Please remember to wear masks and maintain social distancing.",
            "I'm planning my vacation to the beach next summer.",
            "The coronavirus variant shows increased transmission rates.",
            "Hospital ICU capacity has reached critical levels.",
            "Vaccination campaigns have been successful in reducing symptoms.",
            "I love watching movies with my family on weekends.",
            "The lockdown measures have helped flatten the curve.",
            "Research shows proper ventilation reduces virus transmission.",
            "My favorite restaurant finally reopened after restrictions.",
            "Contact tracing has been essential for controlling outbreaks.",
            "The economic impact affects small businesses heavily.",
            "I'm looking forward to traveling again when it's safe.",
            "Health workers deserve recognition for their dedication.",
            "The development of mRNA vaccines was remarkable.",
            "Online learning became the new normal during closures.",
            "Loss of taste and smell are common infection indicators.",
            "Public health policies must balance safety with economics.",
            "I miss going to concerts and large gatherings."
        ]
        
        return pd.DataFrame({'text': sample_texts})
    
    def _generate_simple_embeddings(self) -> np.ndarray:
        """Generar embeddings simples basados en características del texto"""
        embeddings = []
        
        covid_keywords = ['covid', 'coronavirus', 'pandemic', 'virus', 'health', 
                         'disease', 'symptoms', 'vaccination', 'mask', 'lockdown']
        
        for text in self.corpus_data['text']:
            text_lower = text.lower()
            
            # Características básicas
            features = [
                len(text),  # Longitud del texto
                len(text.split()),  # Número de palabras
                sum(1 for keyword in covid_keywords if keyword in text_lower),  # Keywords COVID
                text_lower.count('!'),  # Exclamaciones
                text_lower.count('?'),  # Preguntas
                len([w for w in text.split() if w.isupper()]),  # Palabras en mayúsculas
                text_lower.count('hospital'),  # Menciones de hospital
                text_lower.count('vaccine'),  # Menciones de vacuna
                1 if any(word in text_lower for word in ['i', 'me', 'my']) else 0,  # Texto personal
                1 if any(word in text_lower for word in ['study', 'research', 'data']) else 0  # Texto científico
            ]
            
            embeddings.append(features)
        
        return np.array(embeddings)
    
    def method_1_random_sampling(self, n: int) -> Tuple[List[int], Dict[str, Any]]:
        """Método 1: Muestreo aleatorio simple"""
        indices = random.sample(range(len(self.corpus_data)), min(n, len(self.corpus_data)))
        
        # Métricas de diversidad
        selected_embeddings = self.simple_embeddings[indices]
        metrics = self._calculate_diversity_metrics(selected_embeddings)
        
        return indices, metrics
    
    def method_2_kmeans_like_clustering(self, n: int) -> Tuple[List[int], Dict[str, Any]]:
        """Método 2: Clustering simple para diversidad"""
        if len(self.simple_embeddings) < n:
            indices = list(range(len(self.simple_embeddings)))
        else:
            # Implementación simple de clustering
            indices = []
            remaining_indices = list(range(len(self.simple_embeddings)))
            
            # Seleccionar primer elemento (más cercano al centroide)
            centroid = np.mean(self.simple_embeddings, axis=0)
            distances = [np.linalg.norm(self.simple_embeddings[i] - centroid) 
                        for i in remaining_indices]
            first_idx = remaining_indices[np.argmin(distances)]
            indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Seleccionar resto maximizando distancia
            while len(indices) < n and remaining_indices:
                best_idx = None
                best_distance = -1
                
                for candidate_idx in remaining_indices:
                    # Calcular distancia mínima a elementos ya seleccionados
                    min_dist = min(np.linalg.norm(self.simple_embeddings[candidate_idx] - 
                                                self.simple_embeddings[selected_idx]) 
                                 for selected_idx in indices)
                    
                    if min_dist > best_distance:
                        best_distance = min_dist
                        best_idx = candidate_idx
                
                if best_idx is not None:
                    indices.append(best_idx)
                    remaining_indices.remove(best_idx)
        
        # Métricas de diversidad
        selected_embeddings = self.simple_embeddings[indices]
        metrics = self._calculate_diversity_metrics(selected_embeddings)
        
        return indices, metrics
    
    def method_3_stratified_sampling(self, n: int) -> Tuple[List[int], Dict[str, Any]]:
        """Método 3: Muestreo estratificado por características"""
        # Crear estratos basados en longitud del texto
        text_lengths = [len(text) for text in self.corpus_data['text']]
        
        # Dividir en 3 estratos: corto, medio, largo
        q1 = np.percentile(text_lengths, 33)
        q2 = np.percentile(text_lengths, 66)
        
        short_indices = [i for i, length in enumerate(text_lengths) if length <= q1]
        medium_indices = [i for i, length in enumerate(text_lengths) if q1 < length <= q2]
        long_indices = [i for i, length in enumerate(text_lengths) if length > q2]
        
        # Muestrear proporcionalmente
        n_short = max(1, n // 3)
        n_medium = max(1, n // 3)
        n_long = n - n_short - n_medium
        
        indices = []
        if short_indices:
            indices.extend(random.sample(short_indices, min(n_short, len(short_indices))))
        if medium_indices:
            indices.extend(random.sample(medium_indices, min(n_medium, len(medium_indices))))
        if long_indices:
            indices.extend(random.sample(long_indices, min(n_long, len(long_indices))))
        
        # Completar si faltan
        while len(indices) < n:
            remaining = set(range(len(self.corpus_data))) - set(indices)
            if not remaining:
                break
            indices.append(random.choice(list(remaining)))
        
        # Métricas de diversidad
        selected_embeddings = self.simple_embeddings[indices]
        metrics = self._calculate_diversity_metrics(selected_embeddings)
        metrics['n_strata'] = 3
        
        return indices[:n], metrics
    
    def _calculate_diversity_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de diversidad"""
        if len(embeddings) < 2:
            return {'avg_pairwise_distance': 0.0, 'min_pairwise_distance': 0.0, 
                   'max_pairwise_distance': 0.0, 'std_pairwise_distance': 0.0}
        
        # Calcular distancias euclidean entre todos los pares
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'avg_pairwise_distance': np.mean(distances),
            'min_pairwise_distance': np.min(distances),
            'max_pairwise_distance': np.max(distances),
            'std_pairwise_distance': np.std(distances)
        }
    
    def generate_structured_prompts(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Generar prompts estructurados basados en índices seleccionados"""
        prompts = []
        
        for i, idx in enumerate(indices):
            if idx < len(self.corpus_data):
                text = self.corpus_data.iloc[idx]['text']
                
                # Generar prompt estructurado
                role = random.choice(self.roles)
                task_description = random.choice(self.task_descriptions)
                modifics = random.sample(self.modifics_pool, random.randint(1, 2))
                modifics_str = ", ".join(modifics)
                
                prompt = {
                    'id': f'prompt_{i}',
                    'corpus_index': idx,
                    'role': role,
                    'task_description': task_description,
                    'modifics': modifics_str,
                    'texto': text,
                    'full_prompt': self._format_structured_prompt(role, task_description, modifics_str, text)
                }
                
                prompts.append(prompt)
        
        return prompts
    
    def _format_structured_prompt(self, role: str, task_description: str, modifics: str, text: str) -> str:
        """Formatear prompt con estructura Role + Task + Modifics"""
        return f"""Role: {role}
Task Description: {task_description}
Modifications: {modifics}

Based on the above role and task, analyze this text for COVID-19 related content:
"{text}"

Please provide a COVID-19 relevance score (0-100) and brief explanation."""
    
    def compare_methods(self, population_size: int = 10) -> Dict[str, Any]:
        """Comparar métodos de generación de poblaciones"""
        print(f"\n🔬 Comparando métodos de generación de poblaciones (n={population_size})")
        print("=" * 70)
        
        results = {}
        
        # Método 1: Muestreo aleatorio
        print("1️⃣ Muestreo aleatorio simple...")
        indices1, metrics1 = self.method_1_random_sampling(population_size)
        results['random'] = {
            'indices': indices1,
            'metrics': metrics1,
            'prompts': self.generate_structured_prompts(indices1)
        }
        
        # Método 2: Clustering simple
        print("2️⃣ Clustering para diversidad...")
        indices2, metrics2 = self.method_2_kmeans_like_clustering(population_size)
        results['clustering'] = {
            'indices': indices2,
            'metrics': metrics2,
            'prompts': self.generate_structured_prompts(indices2)
        }
        
        # Método 3: Muestreo estratificado
        print("3️⃣ Muestreo estratificado...")
        indices3, metrics3 = self.method_3_stratified_sampling(population_size)
        results['stratified'] = {
            'indices': indices3,
            'metrics': metrics3,
            'prompts': self.generate_structured_prompts(indices3)
        }
        
        # Generar reporte
        self._generate_comparison_report(results)
        
        return results
    
    def _generate_comparison_report(self, results: Dict[str, Any]):
        """Generar reporte de comparación"""
        print("\n📊 REPORTE DE COMPARACIÓN DE MÉTODOS")
        print("=" * 50)
        
        # Tabla comparativa
        print(f"{'Método':<12} {'Dist.Prom':<10} {'Dist.Min':<10} {'Dist.Max':<10} {'Desv.Std':<10}")
        print("-" * 52)
        
        for method, data in results.items():
            metrics = data['metrics']
            print(f"{method:<12} {metrics['avg_pairwise_distance']:<10.3f} "
                  f"{metrics['min_pairwise_distance']:<10.3f} "
                  f"{metrics['max_pairwise_distance']:<10.3f} "
                  f"{metrics['std_pairwise_distance']:<10.3f}")
        
        # Ranking de métodos
        print("\n🏆 Ranking de Diversidad (por distancia promedio):")
        sorted_methods = sorted(results.items(), 
                              key=lambda x: x[1]['metrics']['avg_pairwise_distance'], 
                              reverse=True)
        
        for i, (method, data) in enumerate(sorted_methods, 1):
            avg_dist = data['metrics']['avg_pairwise_distance']
            print(f"{i}. {method.upper()}: {avg_dist:.3f}")
        
        # Análisis de características
        print("\n📈 ANÁLISIS DE CARACTERÍSTICAS:")
        for method, data in results.items():
            prompts = data['prompts']
            print(f"\n{method.upper()}:")
            print(f"  - Prompts generados: {len(prompts)}")
            print(f"  - Roles únicos: {len(set(p['role'] for p in prompts))}")
            print(f"  - Tasks únicos: {len(set(p['task_description'] for p in prompts))}")
            
            # Mostrar ejemplo de prompt
            if prompts:
                example = prompts[0]
                print(f"  - Ejemplo de texto: '{example['texto'][:60]}...'")
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES:")
        best_method = sorted_methods[0][0]
        print(f"• Mejor método para diversidad: {best_method.upper()}")
        print(f"• Distancia promedio: {sorted_methods[0][1]['metrics']['avg_pairwise_distance']:.3f}")
        
        # Guardar reporte
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'corpus_size': len(self.corpus_data),
            'population_size': len(results['random']['indices']),
            'methods_compared': list(results.keys()),
            'best_method': best_method,
            'results': results
        }
        
        with open('simple_population_study_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Reporte guardado en: simple_population_study_report.json")
    
    def export_population_examples(self, results: Dict[str, Any], output_file: str = 'population_examples.json'):
        """Exportar ejemplos de poblaciones generadas"""
        export_data = {}
        
        for method, data in results.items():
            prompts = data['prompts']
            export_data[method] = {
                'metrics': data['metrics'],
                'sample_prompts': prompts[:3],  # Primeros 3 prompts como ejemplo
                'population_size': len(prompts)
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📋 Ejemplos de poblaciones exportados a: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Estudio Simplificado de Poblaciones Iniciales')
    parser.add_argument('--corpus', default='data/corpus.csv', help='Archivo de corpus CSV')
    parser.add_argument('--population-size', type=int, default=10, help='Tamaño de población')
    parser.add_argument('--sample-size', type=int, default=100, help='Tamaño máximo del corpus')
    parser.add_argument('--export-examples', action='store_true', help='Exportar ejemplos de poblaciones')
    
    args = parser.parse_args()
    
    print("🔬 Iniciando Estudio Simplificado de Poblaciones Iniciales")
    print(f"📊 Parámetros: población={args.population_size}, muestra={args.sample_size}")
    
    # Inicializar estudio
    study = SimplePopulationStudy(
        corpus_file=args.corpus,
        sample_size=args.sample_size
    )
    
    # Ejecutar comparación
    results = study.compare_methods(args.population_size)
    
    if args.export_examples:
        study.export_population_examples(results)
    
    print("\n🎉 Estudio completado!")
    print("📁 Archivos generados:")
    print("  - simple_population_study_report.json")
    if args.export_examples:
        print("  - population_examples.json") 