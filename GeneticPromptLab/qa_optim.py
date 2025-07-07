import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import string
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from .utils import send_query_to_ollama
from .function_templates import function_templates
from ssh_connection import LLMRemoteExecutor
import warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QuestionsAnswersOptimizer:
    def __init__(self, problem_description="", train_questions_list=None, 
                 train_answers_label=None, test_questions_list=None, test_answers_label=None, 
                 label_dict=None, model_name="all-MiniLM-L6-v2", sample_p=1.0, 
                 init_and_fitness_sample=10, window_size_init=1, generations=10, 
                 num_retries=1, ssh_credentials="ssh_credentials.json",
                 modelo_llm="llama3.1", temperatura=0.7):
        
        self.num_retries = num_retries
        self.generations = generations
        self.init_and_fitness_sample = init_and_fitness_sample
        self.test_questions_list = test_questions_list or []
        self.test_answers_label = test_answers_label or []
        self.label_dict = label_dict or {}
        self.problem_description = problem_description
        self.window_size_init = window_size_init
        self.modelo_llm = modelo_llm
        self.temperatura = temperatura
        
        # Configurar SSH executor
        self.ssh_executor = LLMRemoteExecutor(ssh_credentials)
        
        # Configurar modelo de embeddings
        self.model = SentenceTransformer(model_name)
        self.sample_p = sample_p
        
        # Procesar datos de entrenamiento
        if train_questions_list and train_answers_label:
            train_indices_list = np.random.choice(
                np.arange(len(train_questions_list)), 
                size=int(len(train_questions_list) * self.sample_p),
                replace=False
            )
            self.train_questions_list = [train_questions_list[i] for i in train_indices_list]
            self.train_answers_label = [train_answers_label[i] for i in train_indices_list]
            self.embeddings = self.model.encode(self.train_questions_list, show_progress_bar=True)
        else:
            self.train_questions_list = []
            self.train_answers_label = []
            self.embeddings = np.array([])
        
        self.already_sampled_indices = set()

    def create_prompts(self, data: List[Dict[str, str]]) -> List[str]:
        """
        Crear prompts basados en los datos de entrenamiento
        """
        if not data:
            return []
        
        data_doubled = data + data
        prompts = []
        
        for i, item in enumerate(data):
            sample = data_doubled[i:i+self.window_size_init]
            sample_prompt = "\n".join([
                f"Question: \"\"\"{item['q']}\"\"\"\nCorrect Label:\"\"\"{item['a']}\"\"\""
                for item in sample
            ])
            
            messages = [
                {
                    "role": "system", 
                    "content": f"Problem Description: {self.problem_description}\n\n"
                               f"{function_templates[0]['description']}\n\n"
                               f"Note: For this task the labels are: "
                               f"{'; '.join([f'{k}. {v}' for k, v in self.label_dict.items()])}"
                }, 
                {
                    "role": "user", 
                    "content": f"Observe the following samples:\n\n{sample_prompt}"
                }
            ]
            
            try:
                response = send_query_to_ollama(
                    self.ssh_executor, messages, function_templates[0],
                    model=self.modelo_llm, temperature=self.temperatura
                )
                prompt = response.get('prompt', '')
                prompts.append(prompt)
            except Exception as e:
                print(f"Error creating prompt {i}: {e}")
                prompts.append(f"Classify the following question based on the problem description: {self.problem_description}")
        
        return prompts

    def generate_init_prompts(self, n: Optional[int] = None) -> List[str]:
        """
        Generar prompts iniciales para el algoritmo genético
        """
        if n is None:
            n = self.init_and_fitness_sample
        
        if not self.train_questions_list:
            # Retornar prompts genéricos si no hay datos
            return [f"Classify the following question based on: {self.problem_description}"] * n
        
        distinct_sample_indices = self.sample_distinct(n)
        data = []
        
        for sample_index in distinct_sample_indices:
            if sample_index < len(self.train_questions_list):
                question = self.train_questions_list[sample_index]
                answer_label = self.train_answers_label[sample_index]
                answer = self.label_dict.get(answer_label, str(answer_label))
                data.append({"q": question, "a": answer})
        
        if not data:
            return [f"Classify the following question based on: {self.problem_description}"] * n
        
        prompts = self.create_prompts(data)
        return prompts

    def sample_distinct(self, n: int) -> List[int]:
        """
        Muestrear índices distintos usando K-means
        """
        if len(self.embeddings) == 0:
            return list(range(min(n, len(self.train_questions_list))))
        
        embeddings = np.copy(self.embeddings)
        
        if len(self.already_sampled_indices) > 0:
            mask = np.ones(len(embeddings), dtype=bool)
            for idx in self.already_sampled_indices:
                if idx < len(mask):
                    mask[idx] = False
            embeddings = embeddings[mask]
        
        if len(embeddings) == 0:
            return list(range(min(n, len(self.train_questions_list))))
        
        n_clusters = min(n, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        closest_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, embeddings
        )
        
        sampled_indices = set(closest_indices.tolist())
        
        while len(sampled_indices) < n and len(sampled_indices) < len(embeddings):
            remaining_indices = set(range(len(embeddings))) - sampled_indices
            if not remaining_indices:
                break
            
            remaining_indices_array = np.array(list(remaining_indices))
            remaining_embeddings = embeddings[remaining_indices_array]
            n_remaining = min(n - len(sampled_indices), len(remaining_embeddings))
            
            kmeans = KMeans(n_clusters=n_remaining, random_state=0).fit(remaining_embeddings)
            closest_indices, _ = pairwise_distances_argmin_min(
                kmeans.cluster_centers_, remaining_embeddings
            )
            
            sampled_indices.update(closest_indices.tolist())
        
        sampled_indices = list(sampled_indices)[:n]
        self.already_sampled_indices.update(sampled_indices)
        return sampled_indices

    def genetic_algorithm(self, mutation_rate: float = 0.1) -> List[str]:
        """
        Ejecutar algoritmo genético completo
        """
        output_directory = "runs"
        run_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        run_path = os.path.join(output_directory, run_id)
        
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        
        print(f"Run ID: {run_id} has been created at {run_path}")
        
        initial_prompts = self.generate_init_prompts()
        population = initial_prompts
        
        bar = tqdm(range(self.generations))
        
        for gen_id in bar:
            print("Complete Population:", population)
            
            fitness_results = self.evaluate_fitness(population)
            fitness_scores = fitness_results[0]
            questions_list = fitness_results[1]
            correct_answers_list = fitness_results[2]
            prompt_answers_list = fitness_results[3]
            
            top_results = self.select_top_prompts(
                fitness_scores, population, prompt_answers_list
            )
            top_prompts = top_results[0]
            top_prompts_answers_list = top_results[1]
            
            # Guardar resultados
            df = pd.DataFrame({
                'Prompt': population,
                'Fitness Score': fitness_scores
            })
            df.to_csv(os.path.join(run_path, f'epoch_{gen_id}.csv'), index=False)
            
            print("Top Population:", top_prompts)
            print("\n\n")
            
            # Generar nueva población
            new_prompts = self.crossover_using_gpt(
                top_prompts, questions_list, correct_answers_list, 
                top_prompts_answers_list
            )
            
            num_random_prompts = int(self.init_and_fitness_sample * 0.25)
            random_prompts = self.generate_init_prompts(num_random_prompts)
            
            population = top_prompts + new_prompts + random_prompts
            population = self.mutate_prompts(population, mutation_rate)
            
            bar.set_description(str({
                "epoch": gen_id + 1, 
                "acc": round(float(np.mean(fitness_scores)) * 100, 1)
            }))
        
        bar.close()
        return population

    def evaluate_fitness(self, prompts: List[str]) -> Tuple[List[float], List[str], List[str], List[List[str]]]:
        """
        Evaluar fitness de los prompts
        """
        if not self.train_questions_list:
            return [0.0] * len(prompts), [], [], []
        
        distinct_sample_indices = self.sample_distinct(self.init_and_fitness_sample)
        
        just_questions_list = [
            self.train_questions_list[idx] for idx in distinct_sample_indices
            if idx < len(self.train_questions_list)
        ]
        
        questions_list = "\n\n".join([
            f'{i+1}. """{self.train_questions_list[idx]}"""'
            for i, idx in enumerate(distinct_sample_indices)
            if idx < len(self.train_questions_list)
        ])
        
        correct_answers_list = [
            self.label_dict.get(self.train_answers_label[idx], str(self.train_answers_label[idx]))
            for idx in distinct_sample_indices
            if idx < len(self.train_answers_label)
        ]
        
        acc_list = []
        prompt_latest_answers_list = []
        
        for prompt in prompts:
            acc = []
            latest_labels = []
            
            for retry_id in range(self.num_retries):
                try:
                    messages = [
                        {"role": "system", "content": prompt}, 
                        {
                            "role": "user", 
                            "content": f"Questions:\n\n{questions_list}\n\n"
                                      f"Note: Ensure you respond with {len(distinct_sample_indices)} labels."
                        }
                    ]
                    
                    # Preparar function template
                    tmp_function_template = function_templates[1].copy()
                    tmp_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = [
                        v for _, v in self.label_dict.items()
                    ]
                    tmp_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str([
                        v for _, v in self.label_dict.items()
                    ])
                    tmp_function_template["parameters"]["properties"]["label_array"]["minItems"] = len(distinct_sample_indices)
                    tmp_function_template["parameters"]["properties"]["label_array"]["maxItems"] = len(distinct_sample_indices)
                    
                    response = send_query_to_ollama(
                        self.ssh_executor, messages, tmp_function_template,
                        model=self.modelo_llm, temperature=self.temperatura
                    )
                    
                    if 'label_array' in response:
                        labels = [item['label'] for item in response['label_array']]
                    else:
                        labels = []
                    
                    # Asegurar que tenemos el número correcto de labels
                    while len(labels) < len(correct_answers_list):
                        labels.append("unknown")
                    
                    labels = labels[:len(correct_answers_list)]
                    
                    accuracy = sum(1 if a == b else 0 for a, b in zip(labels, correct_answers_list)) / len(labels)
                    acc.append(accuracy)
                    latest_labels = labels
                    
                except Exception as e:
                    print(f"Error evaluating prompt: {e}")
                    acc.append(0.0)
                    latest_labels = ["error"] * len(correct_answers_list)
            
            acc_list.append(sum(acc) / len(acc))
            prompt_latest_answers_list.append(latest_labels)
        
        return acc_list, just_questions_list, correct_answers_list, prompt_latest_answers_list

    def select_top_prompts(self, fitness_scores: List[float], population: List[str], 
                          prompt_answers_list: List[List[str]], 
                          top_fraction: float = 0.5) -> Tuple[List[str], List[List[str]]]:
        """
        Seleccionar mejores prompts basado en fitness
        """
        if not prompt_answers_list:
            prompt_answers_list = [[] for _ in population]
        
        paired_list = list(zip(population, fitness_scores, prompt_answers_list))
        sorted_prompts = sorted(paired_list, key=lambda x: x[1], reverse=True)
        cutoff = int(len(sorted_prompts) * top_fraction)
        
        top_prompts = [prompt for prompt, score, answers_list in sorted_prompts[:cutoff]]
        top_answers = [answers_list for prompt, score, answers_list in sorted_prompts[:cutoff]]
        
        return top_prompts, top_answers

    def crossover_using_gpt(self, prompts: List[str], questions_list: List[str], 
                           correct_answers_list: List[str], 
                           top_prompts_answers_list: List[List[str]]) -> List[str]:
        """
        Crossover usando Ollama para combinar prompts
        """
        if len(prompts) < 2:
            return prompts
        
        new_prompts = []
        
        for i in range(0, len(prompts), 2):
            if i + 1 < len(prompts):
                template = prompts[i]
                additive = prompts[i + 1]
                
                # Obtener respuestas de ambos padres
                answers_from_parents = []
                if i < len(top_prompts_answers_list):
                    answers_from_parents.append(top_prompts_answers_list[i])
                if i + 1 < len(top_prompts_answers_list):
                    answers_from_parents.append(top_prompts_answers_list[i + 1])
                
                # Si los prompts son idénticos, mutar uno
                if template.lower().strip() == additive.lower().strip():
                    additive = self.ollama_mutate(additive)
                
                new_prompt = self.ollama_mix_and_match(
                    template, additive, questions_list, correct_answers_list, 
                    answers_from_parents
                )
                new_prompts.append(new_prompt)
        
        return new_prompts
    
    def ollama_mutate(self, prompt: str) -> str:
        """
        Mutar prompt usando Ollama
        """
        try:
            tmp_function_template = function_templates[2].copy()
            tmp_function_template["parameters"]["properties"]["mutated_prompt"]["description"] += str(round(random.random(), 3))
            
            messages = [
                {
                    "role": "system", 
                    "content": f"You are a prompt-mutator as part of an over-all genetic algorithm. "
                               f"Mutate the following prompt while not detracting from the core-task "
                               f"but still rephrasing/mutating the prompt.\n\n"
                               f"Note: For this task the over-arching Problem Description is: {self.problem_description}"
                }, 
                {
                    "role": "user", 
                    "content": f"Modify the following prompt: \"\"\"{prompt}\"\"\""
                }
            ]
            
            response = send_query_to_ollama(
                self.ssh_executor, messages, tmp_function_template,
                model=self.modelo_llm, temperature=random.random()/2+0.5
            )
            
            mutated_prompt = response.get('mutated_prompt', prompt)
            return mutated_prompt
            
        except Exception as e:
            print(f"Error mutating prompt: {e}")
            return prompt

    def ollama_mix_and_match(self, template: str, additive: str, questions_list: List[str], 
                         correct_answers_list: List[str], 
                         answers_from_parent_prompts: List[List[str]]) -> str:
        """
        Combinar dos prompts usando Ollama
        """
        try:
            # Crear ejemplo con las primeras 5 preguntas
            example_parts = []
            for i in range(min(5, len(questions_list))):
                if i < len(correct_answers_list):
                    example_part = f"Question: \"\"\"{questions_list[i]}\"\"\"\n"
                    example_part += f"Ideal Answer: \"\"\"{correct_answers_list[i]}\"\"\"\n"
                    
                    if len(answers_from_parent_prompts) >= 2:
                        if i < len(answers_from_parent_prompts[0]):
                            example_part += f"Your template parent's answer: \"\"\"{answers_from_parent_prompts[0][i]}\"\"\"\n"
                        if i < len(answers_from_parent_prompts[1]):
                            example_part += f"Your additive parent's answer: \"\"\"{answers_from_parent_prompts[1][i]}\"\"\"\n"
                    
                    example_parts.append(example_part)
            
            example = "\n\n".join(example_parts)
            
            messages = [
                {
                    "role": "system", 
                    "content": f"You are a cross-over system as part of an over-all genetic algorithm. "
                               f"You are to ingrain segments of an additive prompt to that of a template/control "
                               f"prompt to create a healthier offspring.\n\n"
                               f"Note: For this task the over-arching Problem Description is: {self.problem_description}\n\n"
                               f"Example & History for context:{example}\n\n"
                               f"Note: You can use previous mistakes as stepping stones, to quote words/semantics/"
                               f"phrases/keywords/verbs which you think led to the mistake by the AI."
                }, 
                {
                    "role": "user", 
                    "content": f"Template Prompt: \"\"\"{template}\"\"\"\n"
                               f"Additive Prompt: \"\"\"{additive}\"\"\""
                }
            ]
            
            response = send_query_to_ollama(
                self.ssh_executor, messages, function_templates[3],
                model=self.modelo_llm, temperature=self.temperatura
            )
            child_prompt = response.get('child_prompt', template)
            return child_prompt
            
        except Exception as e:
            print(f"Error in crossover: {e}")
            return template

    def mutate_prompts(self, prompts: List[str], mutation_rate: float = 0.1) -> List[str]:
        """
        Mutar prompts basado en tasa de mutación
        """
        mutated_prompts = []
        
        for prompt in prompts:
            if random.random() < mutation_rate:
                mutated_prompts.append(self.ollama_mutate(prompt))
            else:
                mutated_prompts.append(prompt)
        
        return mutated_prompts