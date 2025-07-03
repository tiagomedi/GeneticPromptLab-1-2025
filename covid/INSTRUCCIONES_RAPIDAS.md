# 🚀 Instrucciones Rápidas - COVID-19 Genetic Optimizer

## ⚡ Ejecución Rápida

### 1. Ejecutar optimizador completo (RECOMENDADO)
```bash
python3 run_covid_optimizer.py
```

### 2. Pruebas completas con túnel automático
```bash
python3 test_connection_with_tunnel.py
```

### 3. Solo túnel SSH (mantener activo)
```bash
python3 setup_ssh_tunnel_auto.py
```

### 4. Solo pruebas de conexión
```bash
python3 run_covid_optimizer.py --test-only
```

## 🔧 Solución de Problemas

### Error: "Connection refused"
1. Verificar credenciales en `ssh_credentials.json`
2. Verificar conexión a internet
3. Verificar que el servidor esté accesible

### Error: "pexpect not found"
```bash
pip3 install pexpect
```

### Error: "scikit-learn not found"
```bash
pip3 install scikit-learn
```

## 📋 Pasos Recomendados

1. **Más Simple**: Ejecutar directamente el optimizador
   ```bash
   python3 run_covid_optimizer.py
   ```
   (Maneja automáticamente túnel SSH, pruebas y optimización)

2. **Para diagnóstico**: Si hay problemas, ejecutar pruebas separadas
   ```bash
   python3 test_connection_with_tunnel.py
   ```

3. **Si falla**: Revisar errores y credenciales SSH

## 🔐 Credenciales SSH

Asegúrate de que el archivo `ssh_credentials.json` tenga:
- Credenciales correctas para el jump host
- Credenciales correctas para el servidor target
- Puertos correctos configurados

## 📊 Archivos Principales

- `test_connection_with_tunnel.py` - Pruebas completas con túnel
- `run_covid_optimizer.py` - Ejecutor principal
- `covid_genetic_optimizer.py` - Optimizador genético
- `ssh_credentials.json` - Credenciales SSH
- `corpus.csv` - Datos de entrada (278MB)

## 🎯 Resultado Esperado

Si todo funciona correctamente, deberías ver:
- ✅ Túnel SSH establecido
- ✅ Todas las pruebas pasaron
- ✅ Optimizador genético ejecutándose
- ✅ Resultados guardados en `covid_prompts.json`

## 📊 Ejemplo de ejecución mejorada

```bash
$ python3 run_covid_optimizer.py
🦠 COVID-19 Genetic Prompt Optimizer
==================================================
📋 Configuración del sistema:
   🔌 Puerto SSH: 11435
   📁 Corpus: corpus.csv
   🤖 Modelo: mistral
   👥 Población inicial: 5
   🧬 Generaciones: 3
   📏 Tamaño de muestra: 500

🔧 Configurando entorno...
✅ Dependencias verificadas
🔗 Iniciando túnel SSH integrado...
   💡 Usando credenciales automáticas desde ssh_credentials.json
✅ Túnel SSH configurado exitosamente
✅ Túnel SSH verificado y funcionando
🧪 Probando conexión...
✅ Conexión con Ollama/Mistral verificada
🚀 Ejecutando optimizador genético...
🧬 Iniciando algoritmo genético para 3 generaciones
🎉 ¡Pipeline completado exitosamente!
   Revisa los resultados en: covid_prompts.json

📊 Información del procesamiento:
   📁 Corpus: corpus.csv
   🤖 Modelo: mistral
   👥 Población: 5
   🧬 Generaciones: 3
   📏 Muestra: 500
   🔌 Puerto SSH: 11435

📋 Archivos generados:
   ✅ covid_prompts.json - Prompts optimizados

🔄 Próximos pasos:
   1. Revisar prompts generados en covid_prompts.json
   2. Analizar la efectividad de los prompts
   3. Ejecutar nuevamente con diferentes parámetros si es necesario
🧹 Limpiando recursos...
✅ Túnel SSH cerrado
```