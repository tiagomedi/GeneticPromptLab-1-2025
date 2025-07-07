# SSH Connection Manager para GeneticPromptLab

## Descripción General

Sistema de conexión SSH genérico y reutilizable que permite ejecutar comandos y modelos LLM remotamente a través de jump hosts. Inspirado en el código específico de COVID pero diseñado para ser modular y extensible.

## Características Principales

- **Conexión SSH con Jump Host**: Soporte para conexiones a través de servidores intermedios
- **Gestión Automática de LLM**: Instalación, configuración y ejecución de modelos Ollama
- **Interfaz de Alto Nivel**: Clases fáciles de usar para operaciones comunes
- **Context Managers**: Gestión automática de conexiones
- **Logging Completo**: Seguimiento detallado de operaciones
- **Manejo Robusto de Errores**: Recuperación automática y reintentos
- **Configuración Flexible**: Archivo JSON centralizado

## Archivos del Sistema

### Archivos Principales
- `ssh_connection.py` - Sistema de conexión SSH principal
- `ssh_credentials.json` - Archivo de configuración y credenciales
- `test_ssh_connection.py` - Suite de pruebas completa

### Clases Disponibles

#### `SSHConnectionManager`
Clase principal para gestión de conexiones SSH:
```python
from ssh_connection import SSHConnectionManager

# Uso básico
ssh_manager = SSHConnectionManager()
with ssh_manager.connection_context():
    success, response = ssh_manager.execute_llm_prompt("¿Qué es IA?")
```

#### `LLMRemoteExecutor`
Interfaz de alto nivel para operaciones LLM:
```python
from ssh_connection import LLMRemoteExecutor

# Uso simplificado
executor = LLMRemoteExecutor()
success, response = executor.execute_prompt("Explica machine learning")
```

## Configuración

### Archivo `ssh_credentials.json`

```json
{
  "ssh_config": {
    "jump_host": {
      "host": "200.14.84.16",
      "port": 8080,
      "username": "ignacio.medina1",
      "password": "ignacio.udp2025"
    },
    "target_host": {
      "host": "172.16.40.247",
      "port": 22,
      "username": "colossus",
      "password": "research202x"
    }
  },
  "llm_config": {
    "model": "llama3.1",
    "command": "ollama run llama3.1",
    "service_command": "ollama serve",
    "list_command": "ollama list",
    "install_command": "ollama pull"
  },
  "connection_settings": {
    "command_timeout": 30,
    "llm_startup_delay": 3,
    "llm_response_delay": 8,
    "service_startup_delay": 30,
    "max_retries": 3
  }
}
```

## Instalación

### Dependencias
```bash
pip install paramiko
```

O desde requirements.txt:
```bash
pip install -r requirements.txt
```

### Configuración inicial
1. Copia `ssh_credentials.json` y actualiza las credenciales
2. Ejecuta pruebas: `python test_ssh_connection.py`

## Uso Básico

### 1. Conexión Simple
```python
from ssh_connection import SSHConnectionManager

ssh_manager = SSHConnectionManager()

# Conectar
if ssh_manager.connect():
    # Ejecutar comando
    success, stdout, stderr = ssh_manager.execute_command("whoami")
    print(f"Usuario remoto: {stdout.strip()}")
    
    # Cerrar conexión
    ssh_manager.close()
```

### 2. Context Manager (Recomendado)
```python
from ssh_connection import SSHConnectionManager

ssh_manager = SSHConnectionManager()

with ssh_manager.connection_context():
    # Conexión automática y limpieza
    success, stdout, stderr = ssh_manager.execute_command("ls -la")
    print(stdout)
```

### 3. Ejecución de LLM
```python
from ssh_connection import SSHConnectionManager

ssh_manager = SSHConnectionManager()

with ssh_manager.connection_context():
    # Verificar disponibilidad del LLM
    if ssh_manager.test_llm_availability():
        # Ejecutar prompt
        success, response = ssh_manager.execute_llm_prompt(
            "Explica algoritmos genéticos en una oración."
        )
        if success:
            print(f"Respuesta: {response}")
```

### 4. Interfaz de Alto Nivel
```python
from ssh_connection import LLMRemoteExecutor

executor = LLMRemoteExecutor()

# Probar configuración
if executor.test_setup():
    # Ejecutar prompt simple
    success, response = executor.execute_prompt("¿Qué es inteligencia artificial?")
    
    # Ejecutar batch de prompts
    prompts = ["¿Qué es ML?", "¿Qué es DL?", "¿Qué es NLP?"]
    results = executor.execute_batch_prompts(prompts)
```

## Funcionalidades Avanzadas

### Comandos Interactivos
```python
success, output = ssh_manager.execute_interactive_command(
    "python3",
    inputs=["print('Hello World')", "exit()"],
    read_delay=1.0,
    timeout=30
)
```

### Configuración de Modelos
```python
# Usar modelo específico
success, response = ssh_manager.execute_llm_prompt(
    "Tu prompt aquí",
    model="llama3.1",
    temperature=0.5
)
```

### Logging Personalizado
```python
import logging

# Configurar logging
logger = logging.getLogger('MiApp')
logger.setLevel(logging.DEBUG)

ssh_manager = SSHConnectionManager(logger=logger)
```

## Pruebas y Diagnóstico

### Suite de Pruebas Completa
```bash
python test_ssh_connection.py
```

### Pruebas Individuales
```python
from test_ssh_connection import *

# Probar conexión básica
test_basic_connection()

# Probar disponibilidad LLM
test_llm_availability()

# Probar ejecución de prompts
test_simple_prompt()
```

### Diagnóstico de Problemas

#### Problemas de Conexión
- Verificar credenciales
- Comprobar conectividad de red
- Verificar puertos y firewalls

#### Problemas de LLM
- Verificar que Ollama esté instalado en el servidor remoto
- Comprobar que el modelo esté descargado
- Verificar suficiente memoria en el servidor

## Integración con GeneticPromptLab

### Ejemplo de Uso en Optimización
```python
from ssh_connection import LLMRemoteExecutor
from GeneticPromptLab import QuestionsAnswersOptimizer

# Configurar executor remoto
executor = LLMRemoteExecutor()

class RemoteLLMOptimizer(QuestionsAnswersOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remote_executor = executor
    
    def evaluate_prompt_remotely(self, prompt):
        success, response = self.remote_executor.execute_prompt(prompt)
        return response if success else None
```

## Comparación con /covid

| Aspecto | SSH Connection (Genérico) | /covid (Específico) |
|---------|---------------------------|---------------------|
| **Propósito** | Sistema genérico reutilizable | COVID-19 específico |
| **Modularidad** | Alta, múltiples clases | Acoplado al dominio |
| **Error Handling** | Robusto con logging | Básico |
| **Context Managers** | Sí, automático | Manual |
| **Configuración** | Flexible, JSON detallado | Básica |
| **Testing** | Suite completa | Limitado |
| **Documentación** | Completa | Específica |

## Seguridad

### Mejores Prácticas
- **No hardcodear credenciales** en el código
- **Usar variables de entorno** para datos sensibles
- **Rotar credenciales** regularmente
- **Restringir acceso** al archivo de credenciales

### Configuración Segura
```bash
# Permisos restrictivos
chmod 600 ssh_credentials.json

# Variables de entorno (recomendado)
export SSH_JUMP_HOST="200.14.84.16"
export SSH_JUMP_USER="ignacio.medina1"
# etc.
```

## Troubleshooting

### Errores Comunes

#### `ConnectionError: Failed to establish SSH connection`
- Verificar credenciales
- Comprobar conectividad de red
- Verificar que el jump host esté accesible

#### `❌ Ollama not found on remote server`
- Instalar Ollama en el servidor remoto
- Verificar PATH del usuario remoto

#### `❌ Model llama3.1 not found`
- El sistema intentará instalar automáticamente
- Verificar espacio en disco suficiente
- Comprobar conectividad a internet en el servidor

#### `Timeout en comandos LLM`
- Aumentar `llm_response_delay` en configuración
- Verificar recursos del servidor
- Usar modelos más pequeños si es necesario

### Logs de Diagnóstico
```python
import logging

# Habilitar logging detallado
logging.basicConfig(level=logging.DEBUG)

ssh_manager = SSHConnectionManager()
```

## Desarrollo y Extensión

### Agregar Nuevos Modelos LLM
```python
# En ssh_credentials.json
"llm_config": {
    "model": "mistral",
    "command": "ollama run mistral",
    ...
}
```

### Extender Funcionalidad
```python
class CustomSSHManager(SSHConnectionManager):
    def execute_custom_operation(self):
        # Tu funcionalidad personalizada
        pass
```

## Contribución

Para contribuir al sistema SSH:
1. Mantener compatibilidad con la interfaz existente
2. Agregar tests para nuevas funcionalidades
3. Actualizar documentación
4. Seguir convenciones de logging existentes

## Soporte

Para problemas o preguntas:
1. Ejecutar `python test_ssh_connection.py` para diagnóstico
2. Revisar logs para errores específicos
3. Verificar configuración de red y credenciales 