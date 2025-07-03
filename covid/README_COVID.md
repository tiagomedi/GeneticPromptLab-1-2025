# 🦠 COVID-19 Genetic Prompt Lab

## 📖 **Descripción del proyecto**

Sistema de optimización genética de prompts especializado en análisis de COVID-19, que utiliza:
- **Dataset**: corpus.csv (278MB de datos relacionados con COVID-19)
- **LLM**: Mistral ejecutándose en servidor remoto vía SSH
- **Algoritmo**: Genetic Algorithm para optimización de prompts
- **Muestreo**: TF-IDF + K-Means para diversidad en los datos

## 🚀 **Inicio rápido**

```bash
# 1. Navegar al directorio
cd covid/

# 2. Instalar dependencias
pip install pandas scikit-learn requests numpy

# 3. Ejecutar sistema completo
python run_covid_optimizer.py

# 4. Verificar resultados
cat covid_prompts.json
```

## 🏗️ **Arquitectura del sistema**

### **Componentes principales:**

1. **`run_covid_optimizer.py`** - Script principal que automatiza todo
2. **`covid_genetic_optimizer.py`** - Implementación del algoritmo genético
3. **`setup_ssh_tunnel.py`** - Configuración del túnel SSH
4. **`test_connection.py`** - Verificación de conectividad

### **Flujo de datos:**

```
corpus.csv → TF-IDF → K-Means → Muestreo diverso → Mistral → Prompts genéticos
```

### **Conexión SSH en cadena:**

```
Local:11435 → 200.14.84.16:8080 → 172.16.40.247:11434 → Mistral
```

## 🔧 **Configuración**

### **Credenciales SSH:**
- **Servidor 1**: `ignacio.medina1@200.14.84.16:8080` (contraseña: `xxx`)
- **Servidor 2**: `colossus@172.16.40.247` (contraseña: `xxx`)

### **Parámetros ajustables:**
- `--population`: Tamaño de población inicial (default: 5)
- `--generations`: Número de generaciones (default: 3)
- `--sample-size`: Muestras del corpus a procesar (default: 500)
- `--model`: Modelo LLM a usar (default: mistral)

## 📊 **Funcionalidades específicas de COVID-19**

### **Roles especializados:**
- Healthcare worker
- Public health expert
- Epidemiologist
- Government official
- Researcher
- Policy maker

### **Análisis enfocado en:**
- Impactos de la pandemia
- Medidas de salud pública
- Efectos sociales y económicos
- Políticas de respuesta

### **Salida del sistema:**
- Prompts optimizados para análisis de COVID-19
- Métricas de fitness por generación
- Resultados en formato JSON estructurado

## 🧪 **Comandos útiles**

```bash
# Prueba rápida (3 prompts, 2 generaciones)
python run_covid_optimizer.py --population 3 --generations 2

# Ejecución completa (10 prompts, 10 generaciones)
python run_covid_optimizer.py --population 10 --generations 10

# Solo verificar conexiones
python run_covid_optimizer.py --test-only

# Usar muestra pequeña del corpus
python run_covid_optimizer.py --sample-size 100
```

## 🔍 **Troubleshooting**

### **Error de conexión SSH:**
```bash
# Verificar conexión manual
ssh ignacio.medina1@200.14.84.16 -p 8080
```

### **Mistral no responde:**
```bash
# En el servidor final, verificar Ollama
ollama ps
ollama list
```

### **Corpus muy grande:**
```bash
# Crear versión reducida para pruebas
head -n 1000 corpus.csv > corpus_small.csv
python run_covid_optimizer.py --corpus corpus_small.csv
```

## 📈 **Resultados esperados**

El sistema genera prompts optimizados como:

```json
{
  "prompt": "As a public health expert, analyze the following COVID-19 data to identify patterns in transmission rates and recommend evidence-based intervention strategies...",
  "role": "generated",
  "fitness": 0.85,
  "generation": 3
}
```

## 🔗 **Integración con GeneticPromptLab**

Este módulo extiende la funcionalidad base de GeneticPromptLab:
- Hereda de `GeneticPromptLab` base class
- Implementa métodos específicos para COVID-19
- Utiliza clustering avanzado para diversidad
- Conecta con LLMs remotos vía SSH

## 📁 **Estructura de archivos**

```
covid/
├── run_covid_optimizer.py     # 🚀 Script principal
├── covid_genetic_optimizer.py # 🧬 Algoritmo genético
├── setup_ssh_tunnel.py        # 🔗 Túnel SSH
├── test_connection.py          # 🧪 Pruebas
├── corpus.csv                  # 📊 Dataset (278MB)
├── covid_prompts.json          # 📋 Resultados
├── COMMANDS.md                 # 📚 Comandos
└── README_COVID.md             # 📖 Esta documentación
```

## 🎯 **Próximos pasos**

1. **Optimización completa**: Implementar fitness, selección y cruzamiento avanzados
2. **Métricas avanzadas**: Añadir evaluación semántica de prompts
3. **Visualización**: Crear gráficos de evolución genética
4. **Automatización**: Integrar con pipeline de CI/CD
5. **Escalabilidad**: Soporte para múltiples modelos LLM simultáneos

---

💡 **¿Necesitas ayuda?** Revisa `COMMANDS.md` para comandos detallados o ejecuta `python test_connection.py` para diagnósticos. 
