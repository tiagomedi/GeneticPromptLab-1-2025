
import requests
import json
import random
import re

# URL de la API del LLM
LLM_API_URL = "http://localhost:11434/api/chat"


ROLES = [
    "a normal person",
    "an authority figure",
    "an expert",
    "a healthcare worker",
    "an organization",
    "a policymaker",
]


def seleccionar_rol_aleatorio():
    return random.choice(ROLES)

# se ocupo un exctracto del archivo corpus entregado por el profesor
def cargar_texto_unico(archivo="corpus_reducido_v2.txt"):
    try:
        with open(archivo, "r") as file:
            lineas = [line.strip() for line in file.readlines() if line.strip()]
        if not lineas:
            raise ValueError(f"El archivo {archivo} está vacío o no contiene líneas válidas.")
        return random.choice(lineas)  # Selecciona una línea aleatoria
    except IOError as e:
        raise IOError(f"Error al leer el archivo {archivo}: {e}")

# Función para extraer texto entre comillas dobles de la respuesta
def extraer_contenido_entre_comillas(respuesta):
    return re.findall(r'"(.*?)"', respuesta)


# Función para generar los prompts iniciales
def generar_prompt_para_texto(texto, rol, modelo_llm="llama3.1", temperatura=0.8):
    """
    La ideaes generar un prompt en base al texto de referencia.
    """
    payload = {
        "model": modelo_llm,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that generates structured and narrative prompts for training AI systems. "
                    "Each prompt should have a clear and consistent structure: "
                    "1. Role - Identify the speaker's perspective (e.g., a researcher, an expert, a healthcare worker). "
                    "2. Action - Specify what the speaker is doing (e.g., analyzing, sharing, reporting). "
                    "3. Topic - Provide the main topic (e.g., health issues, pandemic impacts). "
                    "4. Detail (optional) - Add a relevant and meaningful detail to provide context. "
                    "Ensure the generated prompt is concise, meaningful, and aligned with the text provided."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Generate a narrative prompt for the role '{rol}' based on this text: \"{texto}\"."
                )
            }
        ],
        "stream": False,
        "temperature": temperatura
    }

    try:
        response = requests.post(LLM_API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            
            tasks = extraer_contenido_entre_comillas(data.get("message", {}).get("content", ""))
            return tasks[0] if tasks else None
        else:
            print(f"Error al conectarse al LLM: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return None




# Función para guardar resultados en JSON
def guardar_resultados_en_json(prompts, archivo="data.json"):
    try:
        resultados = [
            {"prompt": f"Role: {rol}. Task: {task}", "data_generada": "", "fitness": ""}
            for rol, task in prompts
        ]
        with open(archivo, "w") as file:
            json.dump(resultados, file, indent=4)
        print(f"Resultados guardados en: {archivo}")
    except IOError as e:
        print(f"Error al guardar el archivo JSON: {e}")

# Función principal para generar población inicial
def generar_poblacion_inicial(archivo_json="data.json", cantidad_prompts=10, modelo_llm="llama3", temperatura=1.0):
    # Cargar un texto de referencia automáticamente
    texto_referencia = cargar_texto_unico()
    print(f"Texto de referencia seleccionado: {texto_referencia}")

    prompts_generados = []
    for _ in range(cantidad_prompts):
        rol = seleccionar_rol_aleatorio()
        prompt_generado = None
        intentos = 0

        while not prompt_generado and intentos < 20:
            prompt_generado = generar_prompt_para_texto(texto_referencia, rol, modelo_llm, temperatura)
            intentos += 1
            if not prompt_generado:
                print(f"Intento {intentos}: No se pudo generar un task válido, reintentando...")

        if prompt_generado:
            print(f"Prompt generado exitosamente: Role: {rol}. Task: {prompt_generado}")
            prompts_generados.append((rol, prompt_generado))
        else:
            print("No se pudo generar un task válido después de múltiples intentos.")

    guardar_resultados_en_json(prompts_generados, archivo=archivo_json)

    # Devuelve el texto de referencia usado
    return texto_referencia
