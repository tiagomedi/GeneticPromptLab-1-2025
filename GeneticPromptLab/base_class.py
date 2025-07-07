import pandas as pd
import numpy as np
import random
import logging
import re
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from ssh_connection import LLMRemoteExecutor

class GeneticPromptLab:
    """
    Sistema de optimización genética para prompts de detección COVID usando Ollama vía SSH
    Estructura: Role + Task + Modifics
    """
    
    def __init__(self, corpus_file: str = 'data/corpus.csv', 
                 population_size: int = 10, 
                 generations: int = 5,
                 mutation_rate: float = 0.2,
                 sample_size: int = 1000,
                 modelo_llm: str = 'llama3.1',
                 temperatura: float = 0.7,
                 credentials_file: str = 'ssh_credentials.json'):
        
        self.corpus_file = corpus_file
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.sample_size = sample_size
        self.modelo_llm = modelo_llm
        self.temperatura = temperatura
        self.credentials_file = credentials_file
        
        # Configurar logging
        self.logger = self._setup_logger()
        
        # Configurar executor SSH
        self.ssh_executor = LLMRemoteExecutor(credentials_file)
        
        # Cargar corpus
        self.corpus_data = self._load_corpus()
        
        # Definir componentes de prompt estructurado
        self.roles = [
            "epidemiologist",
            "medical researcher", 
            "public health expert",
            "healthcare analyst",
            "disease surveillance specialist",
            "clinical researcher",
            "health data scientist",
            "medical expert",
            "infectious disease specialist",
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
            "addressing population health metrics",
            "considering socioeconomic health impacts",
            "with emphasis on outbreak detection",
            "highlighting intervention effectiveness",
            "focusing on healthcare resource allocation",
            "considering long-term health outcomes"
        ]
        
        # Keywords para evaluación de fitness
        self.covid_keywords = [
            'covid', 'coronavirus', 'pandemic', 'virus', 'health',
            'disease', 'symptoms', 'vaccination', 'mask', 'social distancing',
            'quarantine', 'lockdown', 'outbreak', 'infection', 'vaccine',
            'fever', 'cough', 'hospital', 'mortality', 'transmission'
        ]
        
        self.logger.info(f"✅ GeneticPromptLab inicializado - Corpus: {len(self.corpus_data)} textos")
    
    def _setup_logger(self) -> logging.Logger:
        """Configurar logger para el sistema"""
        logger = logging.getLogger('GeneticPromptLab')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_corpus(self) -> pd.DataFrame:
        """Cargar corpus de datos"""
        try:
            self.logger.info(f"📄 Cargando corpus desde {self.corpus_file}")
            
            # Cargar con diferentes encodings posibles
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.corpus_file, encoding=encoding)
                    self.logger.info(f"✅ Corpus cargado con encoding {encoding}: {len(df)} registros")
                    
                    # Identificar columna de texto
                    text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'content', 'message', 'tweet', 'post'])]
                    
                    if text_columns:
                        text_col = text_columns[0]
                        self.logger.info(f"📝 Columna de texto detectada: {text_col}")
                        df = df.rename(columns={text_col: 'text'})
                    elif 'text' not in df.columns:
                        # Usar primera columna como texto
                        first_col = str(df.columns[0])
                        df = df.rename(columns={first_col: 'text'})
                        self.logger.warning(f"⚠️ Usando primera columna como texto: {first_col}")
                    
                    # Filtrar textos válidos
                    df = df.dropna(subset=['text'])
                    df = df[df['text'].astype(str).str.len() > 10]  # Mínimo 10 caracteres
                    
                    # Limitar tamaño si es muy grande
                    if len(df) > self.sample_size:
                        df = df.sample(n=self.sample_size, random_state=42)
                        self.logger.info(f"🔄 Corpus limitado a {self.sample_size} registros")
                    
                    return pd.DataFrame(df)
                    
                except UnicodeDecodeError:
                    continue
                    
            raise Exception("No se pudo cargar el corpus con ningún encoding")
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando corpus: {e}")
            
            # Crear datos de ejemplo si no se puede cargar
            self.logger.info("🔄 Creando datos de ejemplo para COVID")
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
            "I miss going to concerts and large gatherings with friends and family."
        ]
        
        return pd.DataFrame({'text': sample_texts})
    
    def create_prompts(self, data: List[str]) -> List[Dict[str, Any]]:
        """
        Crear prompts estructurados usando textos del corpus
        """
        prompts = []
        
        for i, text in enumerate(data):
            # Generar prompt estructurado
            role = random.choice(self.roles)
            task_description = random.choice(self.task_descriptions)
            modifics = random.sample(self.modifics_pool, random.randint(1, 3))
            modifics_str = ", ".join(modifics)
            
            prompt = {
                'id': f'prompt_{i}',
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
    
    def generate_init_prompts(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generar prompts iniciales para el algoritmo genético
        """
        n = n or self.population_size
        
        # Seleccionar textos de referencia del corpus
        reference_texts = self.sample_distinct(n)
        
        # Crear prompts estructurados
        prompts = self.create_prompts(reference_texts)
        
        self.logger.info(f"🧬 Generados {len(prompts)} prompts iniciales")
        return prompts
    
    def sample_distinct(self, n: int) -> List[str]:
        """
        Muestrear textos distintos del corpus
        """
        if len(self.corpus_data) < n:
            # Si el corpus es pequeño, usar con reemplazo
            sample = self.corpus_data.sample(n=n, replace=True, random_state=random.randint(0, 10000))
        else:
            sample = self.corpus_data.sample(n=n, replace=False, random_state=random.randint(0, 10000))
        
        return sample['text'].tolist()
    
    def genetic_algorithm(self, mutation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Ejecutar algoritmo genético completo
        """
        self.logger.info("🚀 Iniciando algoritmo genético COVID")
        
        # Usar mutation_rate del parámetro o del constructor
        mutation_rate = mutation_rate if mutation_rate != 0.1 else self.mutation_rate
        
        # Generar población inicial
        population = self.generate_init_prompts()
        
        # Historial de evolución
        history = []
        
        for generation in range(self.generations):
            self.logger.info(f"🧬 Generación {generation + 1}/{self.generations}")
            
            # Evaluar fitness
            fitness_scores = self.evaluate_fitness(population)
            
            # Registrar estadísticas
            avg_fitness = np.mean(fitness_scores)
            best_fitness = np.max(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            
            history.append({
                'generation': generation + 1,
                'avg_fitness': avg_fitness,
                'best_fitness': best_fitness,
                'best_prompt': population[best_idx].copy()
            })
            
            self.logger.info(f"📊 Fitness promedio: {avg_fitness:.3f}, Mejor: {best_fitness:.3f}")
            
            # Selección de mejores prompts
            top_prompts = self.select_top_prompts(fitness_scores, population, None)
            
            # Crossover y mutación para nueva generación
            if generation < self.generations - 1:
                new_population = self.crossover_using_gpt(top_prompts, None, None, None)
                population = self.mutate_prompts(new_population, mutation_rate)
        
        # Evaluar fitness final
        final_fitness = self.evaluate_fitness(population)
        best_idx = np.argmax(final_fitness)
        
        results = {
            'best_prompt': population[best_idx],
            'best_fitness': final_fitness[best_idx],
            'history': history,
            'final_population': population,
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': mutation_rate,
                'modelo_llm': self.modelo_llm,
                'temperatura': self.temperatura
            }
        }
        
        self.logger.info(f"🎉 Algoritmo completado - Mejor fitness: {final_fitness[best_idx]:.3f}")
        return results
    
    def evaluate_fitness(self, prompts: List[Dict[str, Any]]) -> List[float]:
        """
        Evaluar fitness de prompts usando LLM remoto
        """
        self.logger.info(f"📊 Evaluando fitness de {len(prompts)} prompts")
        
        fitness_scores = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Ejecutar prompt en LLM remoto
                success, response = self.ssh_executor.execute_prompt(
                    prompt['full_prompt'],
                    model=self.modelo_llm,
                    temperature=self.temperatura
                )
                
                if success:
                    # Calcular fitness basado en respuesta
                    fitness = self._calculate_fitness(response, prompt['texto'])
                    fitness_scores.append(fitness)
                    
                    self.logger.debug(f"Prompt {i+1}: fitness={fitness:.3f}")
                else:
                    self.logger.warning(f"❌ Error evaluando prompt {i+1}: {response}")
                    fitness_scores.append(0.0)
                    
            except Exception as e:
                self.logger.error(f"❌ Error evaluando prompt {i+1}: {e}")
                fitness_scores.append(0.0)
        
        return fitness_scores
    
    def _calculate_fitness(self, response: str, original_text: str) -> float:
        """
        Calcular fitness basado en respuesta del LLM
        """
        # Método 1: Buscar puntuación numérica en respuesta
        score_patterns = [
            r'score[:\s]+(\d+)',
            r'(\d+)[/\s]*100',
            r'relevance[:\s]+(\d+)',
            r'(\d+)[:\s]*points?',
            r'(\d+)%'
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    score = int(matches[0])
                    # Normalizar a 0-1
                    return min(score / 100.0, 1.0)
                except ValueError:
                    continue
        
        # Método 2: Análisis de keywords COVID en respuesta
        response_lower = response.lower()
        keyword_matches = sum(1 for keyword in self.covid_keywords if keyword in response_lower)
        
        # Método 3: Análisis de keywords en texto original
        text_lower = original_text.lower()
        text_matches = sum(1 for keyword in self.covid_keywords if keyword in text_lower)
        
        # Combinar métricas
        response_score = min(keyword_matches / 10.0, 1.0)  # Máximo 10 keywords
        text_score = min(text_matches / 5.0, 1.0)  # Máximo 5 keywords
        
        # Fitness combinado
        fitness = (response_score * 0.7) + (text_score * 0.3)
        
        return fitness
    
    def select_top_prompts(self, fitness_scores: List[float], population: List[Dict[str, Any]], 
                          prompt_answers_list: Optional[List] = None, top_fraction: float = 0.5) -> List[Dict[str, Any]]:
        """
        Seleccionar mejores prompts basado en fitness
        """
        n_top = max(1, int(len(population) * top_fraction))
        
        # Ordenar por fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_indices = sorted_indices[:n_top]
        
        top_prompts = [population[i] for i in top_indices]
        
        self.logger.debug(f"🔝 Seleccionados {len(top_prompts)} mejores prompts")
        return top_prompts
    
    def crossover_using_gpt(self, prompts: List[Dict[str, Any]], questions_list: Optional[List] = None,
                           correct_answers_list: Optional[List] = None, 
                           top_prompts_answers_list: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        Crossover: combinar elementos de diferentes prompts
        """
        new_population = []
        
        # Mantener algunos mejores prompts (elitismo)
        elite_count = max(1, len(prompts) // 3)
        new_population.extend(prompts[:elite_count])
        
        # Generar nuevos prompts por crossover
        while len(new_population) < self.population_size:
            # Seleccionar dos padres
            parent1 = random.choice(prompts)
            parent2 = random.choice(prompts)
            
            # Crossover: combinar componentes
            child = self._crossover_structured_prompts(parent1, parent2)
            new_population.append(child)
        
        # Limitar al tamaño de población
        return new_population[:self.population_size]
    
    def _crossover_structured_prompts(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crossover específico para prompts estructurados
        """
        # Combinar componentes de ambos padres
        child_role = random.choice([parent1['role'], parent2['role']])
        child_task = random.choice([parent1['task_description'], parent2['task_description']])
        
        # Combinar modifics
        modifics1 = parent1['modifics'].split(', ')
        modifics2 = parent2['modifics'].split(', ')
        
        # Mezclar modifics de ambos padres
        all_modifics = list(set(modifics1 + modifics2))
        child_modifics = random.sample(all_modifics, random.randint(1, min(3, len(all_modifics))))
        child_modifics_str = ", ".join(child_modifics)
        
        # Seleccionar texto de uno de los padres
        child_text = random.choice([parent1['texto'], parent2['texto']])
        
        child = {
            'id': f'child_{random.randint(1000, 9999)}',
            'role': child_role,
            'task_description': child_task,
            'modifics': child_modifics_str,
            'texto': child_text,
            'full_prompt': self._format_structured_prompt(child_role, child_task, child_modifics_str, child_text)
        }
        
        return child
    
    def mutate_prompts(self, prompts: List[Dict[str, Any]], mutation_rate: float = 0.1) -> List[Dict[str, Any]]:
        """
        Mutar prompts con enfoque en modifics (80% probabilidad)
        """
        mutated_prompts = []
        
        for prompt in prompts:
            if random.random() < mutation_rate:
                mutated_prompt = self._mutate_single_prompt(prompt)
                mutated_prompts.append(mutated_prompt)
            else:
                mutated_prompts.append(prompt)
        
        return mutated_prompts
    
    def _mutate_single_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutar un prompt individual
        """
        mutated = prompt.copy()
        
        # Probabilidades de mutación
        if random.random() < 0.8:  # 80% probabilidad - mutar modifics
            mutated = self._mutate_modifics(mutated)
        
        if random.random() < 0.3:  # 30% probabilidad - mutar role
            mutated['role'] = random.choice(self.roles)
        
        if random.random() < 0.3:  # 30% probabilidad - mutar task
            mutated['task_description'] = random.choice(self.task_descriptions)
        
        # Regenerar prompt completo
        mutated['full_prompt'] = self._format_structured_prompt(
            mutated['role'], 
            mutated['task_description'], 
            mutated['modifics'], 
            mutated['texto']
        )
        
        return mutated
    
    def _mutate_modifics(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutar modifics con diferentes estrategias
        """
        current_modifics = prompt['modifics'].split(', ')
        
        # Estrategias de mutación
        strategy = random.choice(['replace', 'add', 'remove', 'shuffle'])
        
        if strategy == 'replace' and current_modifics:
            # Reemplazar una modificación
            idx = random.randint(0, len(current_modifics) - 1)
            current_modifics[idx] = random.choice(self.modifics_pool)
        
        elif strategy == 'add' and len(current_modifics) < 4:
            # Agregar nueva modificación
            available = [m for m in self.modifics_pool if m not in current_modifics]
            if available:
                current_modifics.append(random.choice(available))
        
        elif strategy == 'remove' and len(current_modifics) > 1:
            # Remover modificación
            current_modifics.pop(random.randint(0, len(current_modifics) - 1))
        
        elif strategy == 'shuffle':
            # Reorganizar modificaciones
            random.shuffle(current_modifics)
        
        prompt['modifics'] = ", ".join(current_modifics)
        return prompt