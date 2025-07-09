#!/usr/bin/env python3
"""
Script para estudiar y comparar diferentes métodos de generación de poblaciones iniciales
para optimización genética de prompts.

Autor: Estudio de Poblaciones Iniciales
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class PopulationStudy:
    """
    Clase para estudiar y comparar métodos de generación de poblaciones iniciales
    """
    
    def __init__(self, corpus_file: str = 'data/corpus.csv', 
                 sample_size: int = 1000,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        
        self.corpus_file = corpus_file
        self.sample_size = sample_size
        self.embedding_model = embedding_model
        
        # Cargar datos
        self.corpus_data = self._load_corpus()
        
        # Configurar modelo de embeddings
        print(f"📊 Cargando modelo de embeddings: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Generar embeddings del corpus
        print(f"🧮 Generando embeddings para {len(self.corpus_data)} textos...")
        self.embeddings = self.model.encode(self.corpus_data['text'].tolist(), 
                                          show_progress_bar=True)
        
        # Componentes para prompts estructurados (COVID)
        self.roles = [
            "epidemiologist", "medical researcher", "public health expert",
            "healthcare analyst", "disease surveillance specialist", "clinical researcher",
            "health data scientist", "medical expert", "infectious disease specialist",
            "health policy analyst"
        ]
        
        self.task_descriptions = [
            "analyzing text content for COVID-19 related information",
            "identifying pandemic-related themes and patterns",
            "detecting health emergency indicators in text",
            "classifying content based on COVID-19 relevance",
            "evaluating text for coronavirus-related keywords",
            "determining pandemic impact in written content",
            "screening text for health crisis indicators",
            "analyzing content for disease outbreak patterns"
        ]
        
        self.modifics_pool = [
            "with focus on symptom identification",
            "emphasizing transmission patterns",
            "highlighting prevention measures",
            "considering healthcare system impact",
            "from a public health surveillance perspective",
            "with attention to policy implications",
            "focusing on community health indicators",
            "incorporating epidemiological evidence",
            "emphasizing risk assessment factors",
            "addressing population health metrics"
        ]
        
        print(f"✅ PopulationStudy inicializado - Corpus: {len(self.corpus_data)} textos")
    
    def _load_corpus(self) -> pd.DataFrame:
        """Cargar corpus de datos"""
        try:
            # Intentar cargar archivo
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.corpus_file, encoding=encoding)
                    print(f"✅ Corpus cargado con encoding {encoding}: {len(df)} registros")
                    
                    # Identificar columna de texto
                    text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'content', 'message', 'tweet', 'post'])]
                    
                    if text_columns:
                        text_col = text_columns[0]
                        df = df.rename(columns={text_col: 'text'})
                    elif 'text' not in df.columns:
                        first_col = str(df.columns[0])
                        df = df.rename(columns={first_col: 'text'})
                    
                    # Filtrar textos válidos
                    df = df.dropna(subset=['text'])
                    df = df[df['text'].astype(str).str.len() > 10]
                    
                    # Limitar tamaño
                    if len(df) > self.sample_size:
                        df = df.sample(n=self.sample_size, random_state=42)
                    
                    return df
                    
                except UnicodeDecodeError:
                    continue
                    
            raise Exception("No se pudo cargar el corpus")
            
        except Exception as e:
            print(f"⚠️ Error cargando corpus: {e}")
            return self._create_sample_covid_data()
    
    def _create_sample_covid_data(self) -> pd.DataFrame:
        """Crear datos de ejemplo para COVID"""
        sample_texts = [
            "I have been experiencing fever, cough, and difficulty breathing for the past week.",
            "The COVID-19 pandemic has significantly impacted global healthcare systems.",
            "Please remember to wear masks and maintain social distancing in public spaces.",
            "I'm planning my vacation to the beach next summer when things get better.",
            "The new coronavirus variant shows increased transmission rates in urban areas.",
            "Hospital ICU capacity has reached critical levels due to COVID-19 admissions.",
            "Vaccination campaigns have been successful in reducing severe COVID symptoms.",
            "I love watching movies and spending time with my family on weekends.",
            "The lockdown measures have helped flatten the curve of infections.",
            "Research shows that proper ventilation reduces virus transmission indoors.",
            "My favorite restaurant finally reopened after the pandemic restrictions.",
            "Contact tracing has been essential for controlling outbreak clusters.",
            "The economic impact of quarantine measures affects small businesses heavily.",
            "I'm looking forward to traveling again once it's safe to do so.",
            "Health workers deserve recognition for their dedication during the crisis.",
            "The development of mRNA vaccines was a remarkable scientific achievement.",
            "Online learning became the new normal during school closures.",
            "Symptoms like loss of taste and smell are common indicators of infection.",
            "Public health policies must balance safety with economic considerations.",
            "I miss going to concerts and large gatherings with friends and family.",
            "The vaccine rollout has been faster than expected in some regions.",
            "Remote work became the norm during the pandemic lockdowns.",
            "Social media misinformation about COVID-19 has been problematic.",
            "The tourism industry was severely affected by travel restrictions.",
            "Mental health support became crucial during isolation periods."
        ]
        
        return pd.DataFrame({'text': sample_texts})
    
    def method_1_random_sampling(self, n: int) -> Tuple[List[int], Dict[str, Any]]:
        """
        Método 1: Muestreo aleatorio simple
        """
        if len(self.corpus_data) < n:
            indices = list(range(len(self.corpus_data)))
            while len(indices) < n:
                indices.extend(random.sample(range(len(self.corpus_data)), 
                                          min(n - len(indices), len(self.corpus_data))))
        else:
            indices = random.sample(range(len(self.corpus_data)), n)
        
        # Métricas de diversidad
        selected_embeddings = self.embeddings[indices]
        diversity_metrics = self._calculate_diversity_metrics(selected_embeddings)
        
        return indices, diversity_metrics
    
    def method_2_kmeans_clustering(self, n: int) -> Tuple[List[int], Dict[str, Any]]:
        """
        Método 2: Clustering K-means para maximizar diversidad
        """
        if len(self.embeddings) == 0:
            return list(range(min(n, len(self.corpus_data)))), {}
        
        embeddings = np.copy(self.embeddings)
        n_clusters = min(n, len(embeddings))
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
        
        # Encontrar puntos más cercanos a centroides
        closest_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, embeddings
        )
        
        indices = closest_indices.tolist()
        
        # Completar si faltan índices
        while len(indices) < n:
            remaining_indices = set(range(len(embeddings))) - set(indices)
            if not remaining_indices:
                break
            
            # Seleccionar punto más lejano a los ya seleccionados
            remaining_embeddings = embeddings[list(remaining_indices)]
            selected_embeddings = embeddings[indices]
            
            distances = cosine_similarity(remaining_embeddings, selected_embeddings)
            min_distances = np.min(distances, axis=1)
            farthest_idx = np.argmax(min_distances)
            
            indices.append(list(remaining_indices)[farthest_idx])
        
        indices = indices[:n]
        
        # Métricas de diversidad
        selected_embeddings = self.embeddings[indices]
        diversity_metrics = self._calculate_diversity_metrics(selected_embeddings)
        diversity_metrics['silhouette_score'] = silhouette_score(embeddings, kmeans.labels_)
        
        return indices, diversity_metrics
    
    def method_3_stratified_sampling(self, n: int) -> Tuple[List[int], Dict[str, Any]]:
        """
        Método 3: Muestreo estratificado basado en clusters
        """
        # Crear clusters para estratificación
        n_strata = min(10, len(self.embeddings) // 10)  # Máximo 10 estratos
        if n_strata < 2:
            return self.method_1_random_sampling(n)
        
        kmeans = KMeans(n_clusters=n_strata, random_state=42).fit(self.embeddings)
        labels = kmeans.labels_
        
        # Muestrear proporcionalmente de cada estrato
        indices = []
        samples_per_stratum = n // n_strata
        
        for stratum in range(n_strata):
            stratum_indices = np.where(labels == stratum)[0]
            
            if len(stratum_indices) > 0:
                # Muestrear del estrato
                n_samples = min(samples_per_stratum, len(stratum_indices))
                selected = np.random.choice(stratum_indices, n_samples, replace=False)
                indices.extend(selected.tolist())
        
        # Completar si faltan muestras
        while len(indices) < n:
            remaining = set(range(len(self.corpus_data))) - set(indices)
            if not remaining:
                break
            indices.append(random.choice(list(remaining)))
        
        indices = indices[:n]
        
        # Métricas de diversidad
        selected_embeddings = self.embeddings[indices]
        diversity_metrics = self._calculate_diversity_metrics(selected_embeddings)
        diversity_metrics['n_strata'] = n_strata
        
        return indices, diversity_metrics
    
    def method_4_maximal_marginal_relevance(self, n: int, lambda_param: float = 0.7) -> Tuple[List[int], Dict[str, Any]]:
        """
        Método 4: Maximal Marginal Relevance (MMR)
        Balancea relevancia y diversidad
        """
        if len(self.embeddings) == 0:
            return list(range(min(n, len(self.corpus_data)))), {}
        
        # Centroide del corpus como "consulta"
        query_embedding = np.mean(self.embeddings, axis=0)
        
        # Similitud con consulta
        relevance_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        indices = []
        remaining_indices = set(range(len(self.embeddings)))
        
        # Seleccionar primer elemento (más relevante)
        first_idx = np.argmax(relevance_scores)
        indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Seleccionar resto usando MMR
        while len(indices) < n and remaining_indices:
            best_score = -np.inf
            best_idx = None
            
            for idx in remaining_indices:
                # Relevancia
                relevance = relevance_scores[idx]
                
                # Similitud máxima con elementos ya seleccionados
                if len(indices) > 0:
                    selected_embeddings = self.embeddings[indices]
                    similarities = cosine_similarity([self.embeddings[idx]], selected_embeddings)[0]
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0
                
                # Puntuación MMR
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # Métricas de diversidad
        selected_embeddings = self.embeddings[indices]
        diversity_metrics = self._calculate_diversity_metrics(selected_embeddings)
        diversity_metrics['lambda_param'] = lambda_param
        
        return indices, diversity_metrics
    
    def _calculate_diversity_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de diversidad"""
        if len(embeddings) < 2:
            return {'avg_pairwise_distance': 0.0, 'min_pairwise_distance': 0.0, 
                   'max_pairwise_distance': 0.0, 'std_pairwise_distance': 0.0}
        
        # Distancias coseno
        similarities = cosine_similarity(embeddings)
        
        # Obtener triángulo superior (sin diagonal)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        distances = 1 - upper_triangle  # Convertir similitud a distancia
        
        return {
            'avg_pairwise_distance': np.mean(distances),
            'min_pairwise_distance': np.min(distances),
            'max_pairwise_distance': np.max(distances),
            'std_pairwise_distance': np.std(distances)
        }
    
    def generate_structured_prompts(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Generar prompts estructurados basados en índices seleccionados
        """
        prompts = []
        
        for i, idx in enumerate(indices):
            if idx < len(self.corpus_data):
                text = self.corpus_data.iloc[idx]['text']
                
                # Generar prompt estructurado
                role = random.choice(self.roles)
                task_description = random.choice(self.task_descriptions)
                modifics = random.sample(self.modifics_pool, random.randint(1, 3))
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
    
    def compare_methods(self, population_size: int = 20) -> Dict[str, Any]:
        """
        Comparar todos los métodos de generación de poblaciones
        """
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
        
        # Método 2: K-means clustering
        print("2️⃣ K-means clustering...")
        indices2, metrics2 = self.method_2_kmeans_clustering(population_size)
        results['kmeans'] = {
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
        
        # Método 4: Maximal Marginal Relevance
        print("4️⃣ Maximal Marginal Relevance...")
        indices4, metrics4 = self.method_4_maximal_marginal_relevance(population_size)
        results['mmr'] = {
            'indices': indices4,
            'metrics': metrics4,
            'prompts': self.generate_structured_prompts(indices4)
        }
        
        # Generar comparación visual
        self._visualize_comparison(results)
        
        # Generar reporte
        self._generate_comparison_report(results)
        
        return results
    
    def _visualize_comparison(self, results: Dict[str, Any]):
        """Crear visualizaciones para comparar métodos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Métodos de Generación de Poblaciones Iniciales', fontsize=16)
        
        # Métricas de diversidad
        methods = list(results.keys())
        avg_distances = [results[method]['metrics']['avg_pairwise_distance'] for method in methods]
        min_distances = [results[method]['metrics']['min_pairwise_distance'] for method in methods]
        max_distances = [results[method]['metrics']['max_pairwise_distance'] for method in methods]
        std_distances = [results[method]['metrics']['std_pairwise_distance'] for method in methods]
        
        # Gráfico 1: Distancias promedio
        axes[0, 0].bar(methods, avg_distances, alpha=0.7, color=['blue', 'green', 'red', 'purple'])
        axes[0, 0].set_title('Distancia Promedio Entre Elementos')
        axes[0, 0].set_ylabel('Distancia Coseno')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Rango de distancias
        x_pos = np.arange(len(methods))
        axes[0, 1].bar(x_pos, max_distances, alpha=0.7, label='Máxima', color='lightcoral')
        axes[0, 1].bar(x_pos, min_distances, alpha=0.7, label='Mínima', color='lightblue')
        axes[0, 1].set_title('Rango de Distancias')
        axes[0, 1].set_ylabel('Distancia Coseno')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(methods, rotation=45)
        axes[0, 1].legend()
        
        # Gráfico 3: Desviación estándar
        axes[1, 0].bar(methods, std_distances, alpha=0.7, color=['orange', 'cyan', 'pink', 'yellow'])
        axes[1, 0].set_title('Variabilidad de Distancias')
        axes[1, 0].set_ylabel('Desviación Estándar')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Gráfico 4: Visualización 2D con PCA
        all_indices = []
        colors = ['blue', 'green', 'red', 'purple']
        labels = []
        
        for i, method in enumerate(methods):
            indices = results[method]['indices']
            all_indices.extend(indices)
            labels.extend([method] * len(indices))
        
        if len(all_indices) > 0:
            selected_embeddings = self.embeddings[all_indices]
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(selected_embeddings)
            
            for i, method in enumerate(methods):
                method_mask = np.array(labels) == method
                method_points = pca_result[method_mask]
                axes[1, 1].scatter(method_points[:, 0], method_points[:, 1], 
                                 c=colors[i], label=method, alpha=0.7)
            
            axes[1, 1].set_title('Distribución 2D de Poblaciones (PCA)')
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC2')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('population_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_comparison_report(self, results: Dict[str, Any]):
        """Generar reporte de comparación"""
        print("\n📊 REPORTE DE COMPARACIÓN DE MÉTODOS")
        print("=" * 50)
        
        # Tabla comparativa
        metrics_df = pd.DataFrame({
            method: results[method]['metrics'] for method in results.keys()
        }).T
        
        print("\n🔍 Métricas de Diversidad:")
        print(metrics_df.round(4))
        
        # Ranking de métodos
        print("\n🏆 Ranking de Diversidad (por distancia promedio):")
        ranking = metrics_df.sort_values('avg_pairwise_distance', ascending=False)
        for i, (method, row) in enumerate(ranking.iterrows(), 1):
            print(f"{i}. {method.upper()}: {row['avg_pairwise_distance']:.4f}")
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES:")
        best_method = ranking.index[0]
        print(f"• Mejor método para diversidad: {best_method.upper()}")
        print(f"• Distancia promedio: {ranking.iloc[0]['avg_pairwise_distance']:.4f}")
        
        if 'silhouette_score' in ranking.columns:
            print(f"• Calidad de clustering: {ranking.iloc[0].get('silhouette_score', 'N/A')}")
        
        # Guardar reporte
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'corpus_size': len(self.corpus_data),
            'embedding_model': self.embedding_model,
            'methods_compared': list(results.keys()),
            'metrics': metrics_df.to_dict(),
            'ranking': ranking.index.tolist(),
            'best_method': best_method
        }
        
        with open('population_study_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Reporte guardado en: population_study_report.json")
    
    def run_complete_study(self, population_sizes: List[int] = [10, 20, 50]) -> Dict[str, Any]:
        """
        Ejecutar estudio completo con diferentes tamaños de población
        """
        print("🚀 INICIANDO ESTUDIO COMPLETO DE POBLACIONES INICIALES")
        print("=" * 60)
        
        complete_results = {}
        
        for pop_size in population_sizes:
            print(f"\n📊 Analizando población de tamaño: {pop_size}")
            results = self.compare_methods(pop_size)
            complete_results[f'pop_{pop_size}'] = results
        
        # Análisis de tendencias
        self._analyze_trends(complete_results)
        
        return complete_results
    
    def _analyze_trends(self, complete_results: Dict[str, Any]):
        """Analizar tendencias en diferentes tamaños de población"""
        print("\n📈 ANÁLISIS DE TENDENCIAS")
        print("=" * 30)
        
        # Crear DataFrame para análisis
        trend_data = []
        
        for pop_key, results in complete_results.items():
            pop_size = int(pop_key.split('_')[1])
            
            for method, data in results.items():
                trend_data.append({
                    'population_size': pop_size,
                    'method': method,
                    'avg_distance': data['metrics']['avg_pairwise_distance'],
                    'min_distance': data['metrics']['min_pairwise_distance'],
                    'max_distance': data['metrics']['max_pairwise_distance'],
                    'std_distance': data['metrics']['std_pairwise_distance']
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        # Visualizar tendencias
        plt.figure(figsize=(12, 8))
        
        for method in trend_df['method'].unique():
            method_data = trend_df[trend_df['method'] == method]
            plt.plot(method_data['population_size'], method_data['avg_distance'], 
                    marker='o', label=method, linewidth=2)
        
        plt.title('Tendencia de Diversidad vs Tamaño de Población')
        plt.xlabel('Tamaño de Población')
        plt.ylabel('Distancia Promedio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('diversity_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Guardar análisis de tendencias
        trend_df.to_csv('diversity_trends.csv', index=False)
        print("💾 Análisis de tendencias guardado en: diversity_trends.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Estudio de Métodos de Generación de Poblaciones Iniciales')
    parser.add_argument('--corpus', default='data/corpus.csv', help='Archivo de corpus CSV')
    parser.add_argument('--population-size', type=int, default=20, help='Tamaño de población a estudiar')
    parser.add_argument('--complete-study', action='store_true', help='Ejecutar estudio completo con múltiples tamaños')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', help='Modelo de embeddings')
    parser.add_argument('--sample-size', type=int, default=1000, help='Tamaño máximo del corpus')
    
    args = parser.parse_args()
    
    print("🔬 Iniciando Estudio de Poblaciones Iniciales")
    print(f"📊 Parámetros: corpus={args.corpus}, población={args.population_size}")
    print(f"🧮 Modelo embeddings: {args.embedding_model}")
    
    # Inicializar estudio
    study = PopulationStudy(
        corpus_file=args.corpus,
        sample_size=args.sample_size,
        embedding_model=args.embedding_model
    )
    
    if args.complete_study:
        # Estudio completo
        print("🚀 Ejecutando estudio completo...")
        results = study.run_complete_study([10, 20, 50])
        
        print("\n🎉 Estudio completo finalizado!")
        print("📁 Archivos generados:")
        print("  - population_study_report.json")
        print("  - population_comparison.png")
        print("  - diversity_trends.png")
        print("  - diversity_trends.csv")
        
    else:
        # Estudio simple
        print(f"🔍 Ejecutando comparación para población de {args.population_size}...")
        results = study.compare_methods(args.population_size)
        
        print(f"\n🎉 Comparación completada!")
        print("📁 Archivos generados:")
        print("  - population_study_report.json")
        print("  - population_comparison.png") 