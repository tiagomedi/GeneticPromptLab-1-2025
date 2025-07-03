# 🔗 Túnel SSH Automático para COVID-19 Genetic Optimizer

## ✨ Funcionalidad

Ahora puedes ejecutar el optimizador genético sin necesidad de ingresar manualmente las contraseñas SSH. Las credenciales se almacenan en un archivo JSON y se usan automáticamente.

## 📁 Archivos incluidos

- `ssh_credentials.json` - Credenciales SSH estructuradas
- `setup_ssh_tunnel_auto.py` - Script automático para túnel SSH
- `run_covid_optimizer.py` - Script principal actualizado
- `test_auto_ssh.py` - Script de prueba

## 🚀 Uso

### Opción 1: Pipeline completo (recomendado)
```bash
python3 run_covid_optimizer.py
```

### Opción 2: Solo túnel SSH
```bash
python3 setup_ssh_tunnel_auto.py
```

### Opción 3: Prueba rápida
```bash
python3 test_auto_ssh.py
```

## 🔧 Configuración

### Credenciales SSH (`ssh_credentials.json`)
<!-- ```json
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
  "tunnel_config": {
    "local_port": 11435,
    "remote_host": "172.16.40.247",
    "remote_port": 11434
  }
}
``` -->

## 🔄 Flujo de conexión

1. **Jump Host**: `ignacio.medina1@200.14.84.16:8080`
2. **Target Host**: `colossus@172.16.40.247:22`
3. **Túnel**: `localhost:11435 → 172.16.40.247:11434`
4. **Ollama**: Accesible en `http://localhost:11435/api/chat`

## 📦 Dependencias

El script instala automáticamente `pexpect` si no está disponible.

## 🛠️ Solución de problemas

### Error: "pexpect not found"
```bash
pip3 install pexpect
```

### Error: "Connection refused"
- Verifica que las credenciales estén correctas
- Verifica que el servidor esté accesible
- Verifica que Ollama esté ejecutándose en el servidor

### Error: "Timeout"
- Aumenta el timeout en el script
- Verifica la conexión a internet
- Verifica que los puertos no estén bloqueados

## 🔐 Seguridad

⚠️ **IMPORTANTE**: El archivo `ssh_credentials.json` contiene contraseñas en texto plano. Asegúrate de:
- No incluirlo en el control de versiones
- Restringir permisos de lectura (`chmod 600 ssh_credentials.json`)
- Usar un directorio seguro

## 📊 Ejemplo de ejecución

```bash
$ python3 run_covid_optimizer.py
🦠 COVID-19 Genetic Prompt Optimizer
==================================================
🔧 Configurando entorno...
✅ Dependencias verificadas
🔗 Iniciando túnel SSH automático...
   💡 Usando credenciales automáticas desde ssh_credentials.json
   Esperando conexión...
✅ Túnel SSH automático iniciado
🧪 Probando conexión...
✅ Todas las pruebas pasaron
🚀 Ejecutando optimizador genético...
🧬 Iniciando algoritmo genético para 3 generaciones
🎉 ¡Pipeline completado exitosamente!
``` 