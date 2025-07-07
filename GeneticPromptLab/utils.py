import json
import time
import re
from typing import Dict, Any, List
from ssh_connection import LLMRemoteExecutor

def send_query_to_ollama(ssh_executor: LLMRemoteExecutor, messages: List[Dict[str, str]], 
                        function_template: Dict[str, Any], 
                        model: str = 'llama3.1',
                        temperature: float = 0.7,
                        pause: int = 2) -> Dict[str, Any]:
    """
    Enviar queries a Ollama vía SSH
    """
    # Convertir messages a prompt único
    prompt = format_messages_to_prompt(messages, function_template)
    
    # Ejecutar en LLM remoto
    success, response = ssh_executor.execute_prompt(
        prompt,
        model=model,
        temperature=temperature
    )
    
    if not success:
        raise Exception(f"Error ejecutando prompt en Ollama: {response}")
    
    # Extraer respuesta estructurada
    extracted_response = extract_structured_response(response, function_template)
    
    time.sleep(pause)
    return extracted_response

def format_messages_to_prompt(messages: List[Dict[str, str]], function_template: Dict[str, Any]) -> str:
    """
    Convierte mensajes de OpenAI format a prompt único para Ollama
    """
    prompt_parts = []
    
    # Agregar contexto del sistema
    for message in messages:
        if message["role"] == "system":
            prompt_parts.append(f"System: {message['content']}")
        elif message["role"] == "user":
            prompt_parts.append(f"User: {message['content']}")
    
    # Agregar instrucciones del function template
    function_name = function_template["name"]
    function_desc = function_template["description"]
    
    prompt_parts.append(f"\nTask: {function_desc}")
    
    # Agregar formato de respuesta esperado
    if function_name == "generate_prompts":
        prompt_parts.append("\nPlease provide your response in the following format:")
        prompt_parts.append("PROMPT: [your generated prompt here]")
    
    elif function_name == "QnA_bot":
        prompt_parts.append("\nPlease provide your response in the following format:")
        prompt_parts.append("LABELS: [label1, label2, label3, ...]")
    
    elif function_name == "prompt_mutate":
        prompt_parts.append("\nPlease provide your response in the following format:")
        prompt_parts.append("MUTATED_PROMPT: [your mutated prompt here]")
    
    elif function_name == "prompt_crossover":
        prompt_parts.append("\nPlease provide your response in the following format:")
        prompt_parts.append("CHILD_PROMPT: [your child prompt here]")
    
    return "\n\n".join(prompt_parts)

def extract_structured_response(response: str, function_template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae respuesta estructurada de la respuesta del LLM
    """
    function_name = function_template["name"]
    
    if function_name == "generate_prompts":
        # Buscar PROMPT: en la respuesta
        match = re.search(r'PROMPT:\s*(.+?)(?:\n|$)', response, re.DOTALL)
        if match:
            return {"prompt": match.group(1).strip()}
        
        # Fallback: usar toda la respuesta
        return {"prompt": response.strip()}
    
    elif function_name == "QnA_bot":
        # Buscar LABELS: en la respuesta
        match = re.search(r'LABELS:\s*\[(.*?)\]', response, re.DOTALL)
        if match:
            labels_str = match.group(1)
            # Parsear labels
            labels = []
            for label in labels_str.split(','):
                clean_label = label.strip().strip('"\'')
                if clean_label:
                    labels.append({"label": clean_label})
            return {"label_array": labels}
        
        # Fallback: buscar labels en formato diferente
        labels = extract_labels_from_response(response)
        return {"label_array": [{"label": label} for label in labels]}
    
    elif function_name == "prompt_mutate":
        # Buscar MUTATED_PROMPT: en la respuesta
        match = re.search(r'MUTATED_PROMPT:\s*(.+?)(?:\n|$)', response, re.DOTALL)
        if match:
            return {"mutated_prompt": match.group(1).strip()}
        
        # Fallback: usar toda la respuesta
        return {"mutated_prompt": response.strip()}
    
    elif function_name == "prompt_crossover":
        # Buscar CHILD_PROMPT: en la respuesta
        match = re.search(r'CHILD_PROMPT:\s*(.+?)(?:\n|$)', response, re.DOTALL)
        if match:
            return {"child_prompt": match.group(1).strip()}
        
        # Fallback: usar toda la respuesta
        return {"child_prompt": response.strip()}
    
    # Fallback genérico
    return {"response": response.strip()}

def extract_labels_from_response(response: str) -> List[str]:
    """
    Extrae labels de la respuesta usando varios patrones
    """
    labels = []
    
    # Patrón 1: Lista numerada
    numbered_pattern = r'^\d+\.\s*(.+?)$'
    for match in re.finditer(numbered_pattern, response, re.MULTILINE):
        label = match.group(1).strip()
        if label:
            labels.append(label)
    
    # Patrón 2: Lista con guiones
    if not labels:
        dash_pattern = r'^-\s*(.+?)$'
        for match in re.finditer(dash_pattern, response, re.MULTILINE):
            label = match.group(1).strip()
            if label:
                labels.append(label)
    
    # Patrón 3: Palabras clave comunes
    if not labels:
        # Buscar palabras clave típicas de clasificación
        keywords = ['business', 'sports', 'world', 'science', 'technology', 'health', 'entertainment']
        for keyword in keywords:
            if keyword.lower() in response.lower():
                labels.append(keyword)
    
    return labels