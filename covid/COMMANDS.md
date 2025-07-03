# 🦠 COVID-19 Genetic Prompt Lab - Comandos de Ejecución

## 📋 Pasos para ejecutar el sistema completo

### 🚀 **Opción 1: Ejecución automática (RECOMENDADO)**
```bash
cd covid/
python run_covid_optimizer.py

# O con parámetros personalizados:
python run_covid_optimizer.py --population 8 --generations 5 --sample-size 1000
```

### 🔧 **Opción 2: Ejecución manual paso a paso**

#### 1. Configurar túnel SSH
```bash
# Terminal 1: Mantener abierto para el túnel SSH
cd covid/
python setup_ssh_tunnel.py

# Cuando se ejecute, ingresa las contraseñas:
# Primera conexión: ignacio.udp2025
# Segunda conexión: research202x
```

#### 2. Ejecutar optimizador genético
```bash
# Terminal 2: Ejecutar en paralelo
cd covid/
python covid_genetic_optimizer.py
```

### 🧪 **Opción 3: Solo pruebas**
```bash
cd covid/
python test_connection.py

# O usar el runner en modo prueba:
python run_covid_optimizer.py --test-only
```

### 3. 🔍 **Verificar conexión manualmente**
```bash
# Probar conexión SSH paso a paso
ssh ignacio.medina1@200.14.84.16 -p 8080
# Password: ignacio.udp2025

# Dentro del primer servidor:
ssh colossus@172.16.40.247  
# Password: research202x

# Dentro del segundo servidor:
curl http://localhost:11434/api/tags
# Debería mostrar los modelos disponibles
```

### 4. 📊 **Visualizar resultados**
```bash
# Ver prompts generados
cat covid_prompts.json

# Si tienes el visualizador original:
python ../visualizer.py
```

## 🛠️ **Configuración de dependencias**

### Instalar dependencias adicionales:
```bash
pip install pandas scikit-learn requests
```

### Verificar que Mistral esté disponible en el servidor:
```bash
# En el servidor final (172.16.40.247):
ollama list
ollama run mistral
```

## 🔍 **Troubleshooting**

### Si el túnel SSH falla:
```bash
# Verificar conexión paso a paso
ssh -v ignacio.medina1@200.14.84.16 -p 8080

# Probar túnel directo
ssh -L 11435:172.16.40.247:11434 ignacio.medina1@200.14.84.16 -p 8080
```

### Si Mistral no responde:
```bash
# En el servidor final:
ollama ps  # Ver procesos activos
ollama serve  # Iniciar servidor si no está corriendo
```

### Si corpus.csv es muy grande:
```bash
# Crear versión reducida para pruebas
head -n 1000 corpus.csv > corpus_small.csv
```

## 📈 **Flujo de trabajo completo**

1. **Setup inicial**: `python setup_ssh_tunnel.py`
2. **Optimización**: `python covid_genetic_optimizer.py`
3. **Análisis**: Revisar `covid_prompts.json`
4. **Iteración**: Ajustar parámetros y repetir

## 🔧 **Parámetros ajustables**

En `covid_genetic_optimizer.py`:
- `sample_size`: Tamaño del corpus a procesar
- `init_population_size`: Número de prompts iniciales
- `generations`: Número de generaciones genéticas
- `model_name`: Modelo de Mistral a usar

## 📝 **Estructura de archivos**

```
covid/
├── run_covid_optimizer.py   # 🚀 Script principal (USAR ESTE)
├── setup_ssh_tunnel.py      # Configuración del túnel SSH
├── covid_genetic_optimizer.py  # Optimizador genético especializado
├── test_connection.py        # Pruebas de conectividad
├── prueba.py                # Script original de referencia
├── corpus.csv               # Dataset de COVID-19 (278MB)
├── covid_prompts.json       # Resultados generados
└── COMMANDS.md              # Este archivo
```

## 🎛️ **Parámetros disponibles**

### Para `run_covid_optimizer.py`:
```bash
python run_covid_optimizer.py [opciones]

--ssh-port PORT      Puerto local para túnel SSH (default: 11435)
--corpus FILE        Archivo del corpus (default: corpus.csv)
--model MODEL        Modelo de LLM (default: mistral)
--population N       Tamaño de población inicial (default: 5)
--generations N      Número de generaciones (default: 3)
--sample-size N      Tamaño de muestra del corpus (default: 500)
--test-only          Solo ejecutar pruebas de conexión
```

### Ejemplos de uso:
```bash
# Ejecución rápida para pruebas
python run_covid_optimizer.py --population 3 --generations 2

# Ejecución completa
python run_covid_optimizer.py --population 10 --generations 10 --sample-size 2000

# Solo probar conexiones
python run_covid_optimizer.py --test-only
``` 