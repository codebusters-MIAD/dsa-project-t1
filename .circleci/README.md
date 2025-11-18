# CircleCI Pipeline Configuration

## Estructura de Pipelines

Este proyecto utiliza una arquitectura modular de pipelines en CircleCI con archivos separados para mejor mantenibilidad.

### Archivos

```
.circleci/
├── config.yml                    # Configuracion principal con path filtering
└── pipelines/
    └── dashboard.yml             # Pipeline del dashboard
```

## Pipeline del Dashboard

### Descripcion

Pipeline automatizado para testing, building y deployment del dashboard de FilmLens a Railway.

### Triggers

El pipeline se ejecuta cuando hay cambios en:
- `src/dashboard/**`
- `docker/Dockerfile.dashboard`

### Jobs

1. **test-dashboard**
   - Instala dependencias de Python
   - Ejecuta linting con flake8
   - Valida imports del dashboard
   - Verifica estructura de archivos

2. **build-dashboard-docker**
   - Construye imagen Docker del dashboard
   - Prueba el container localmente
   - Persiste informacion de la imagen

3. **deploy-dashboard-railway**
   - Instala Railway CLI
   - Despliega el dashboard a Railway
   - Verifica que el deployment sea exitoso

4. **notify-deployment**
   - Notifica estado del deployment

### Workflow

```
test-dashboard
    ↓
build-dashboard-docker
    ↓
deploy-dashboard-railway (solo en branch main)
    ↓
notify-deployment
```

## Variables de Entorno Requeridas

Configura estas variables en CircleCI (Project Settings → Environment Variables):

- `RAILWAY_TOKEN`: Token de autenticacion de Railway
- `RAILWAY_DASHBOARD_URL`: URL del dashboard en Railway (opcional)

## Obteniendo Railway Token

1. Instala Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login en Railway:
   ```bash
   railway login
   ```

3. Obtener token:
   ```bash
   railway whoami --token
   ```

4. Agregar token a CircleCI como variable de entorno `RAILWAY_TOKEN`

## Configuracion en Railway

### Opcion 1: Usando Railway CLI

```bash
# Crear nuevo proyecto
railway init

# Crear servicio para dashboard
railway service create dashboard

# Configurar variables de entorno
railway variables set API_BASE_URL=https://your-api-url.com

# Deploy manual inicial
railway up --service dashboard
```

### Opcion 2: Usando Railway Dashboard

1. Crear nuevo proyecto en Railway
2. Conectar repositorio de GitHub
3. Configurar:
   - **Service Name**: dashboard
   - **Root Directory**: ./
   - **Dockerfile Path**: docker/Dockerfile.dashboard
   - **Port**: 8050

4. Agregar variables de entorno:
   - `API_BASE_URL`: URL de tu API de FilmLens

## Testing Local

### Test del pipeline localmente

```bash
# Instalar CircleCI CLI
curl -fLSs https://circle.ci/cli | bash

# Validar configuracion
circleci config validate

# Ejecutar job localmente
circleci local execute --job test-dashboard
```

### Test del dashboard

```bash
cd src/dashboard

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar linting
flake8 . --max-line-length=120

# Validar imports
python -c "import app"

# Correr dashboard
python app.py
```

### Test del Docker build

```bash
# Build imagen
docker build -f docker/Dockerfile.dashboard -t filmlens-dashboard:test .

# Correr container
docker run -p 8050:8050 -e API_BASE_URL=http://localhost:8000 filmlens-dashboard:test

# Verificar
curl http://localhost:8050
```

## Monitoreo del Pipeline

### Ver status del pipeline

1. Ve a CircleCI dashboard
2. Selecciona tu proyecto `dsa-project-t1`
3. Ve la lista de workflows y jobs

### Ver logs del deployment

```bash
# Logs de Railway
railway logs --service dashboard

# Status del servicio
railway status
```

## Troubleshooting

### Error: Path filtering no funciona

**Solucion**: Verifica que el setup workflow este configurado correctamente y que hayas habilitado "Enable dynamic config using setup workflows" en Project Settings.

### Error: Railway deployment falla

**Solucion**: 
1. Verifica que `RAILWAY_TOKEN` este configurado
2. Asegurate que el servicio exista en Railway
3. Revisa logs con `railway logs`

### Error: Docker build falla

**Solucion**:
1. Verifica que `docker/Dockerfile.dashboard` exista
2. Revisa que todas las dependencias esten en `requirements.txt`
3. Prueba el build localmente primero

### Error: Health check falla

**Solucion**:
1. Verifica que Railway este sirviendo en el puerto correcto
2. Asegurate que las variables de entorno esten configuradas
3. Revisa los logs del container en Railway

## Siguientes Pasos

1. Agregar mas pipelines para otros componentes:
   - `pipelines/api.yml`
   - `pipelines/query-api.yml`
   - `pipelines/model-training.yml`

2. Agregar tests unitarios al dashboard

3. Implementar notificaciones (Slack, email)

4. Agregar metrics y monitoring
