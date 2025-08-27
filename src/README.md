# Código fuente de la aplicación (src)

Esta carpeta contiene el código fuente principal de la solución de agentes de IA, incluyendo la API backend, el frontend y los recursos de datos.

## Estructura

- **api/**: Backend de la aplicación, implementado en Python (FastAPI). Expone los endpoints REST y gestiona la comunicación con los servicios de Azure AI, almacenamiento, búsqueda, etc.
  - `main.py`: Punto de entrada de la API.
  - `routes.py`: Define las rutas/endpoints y la lógica de negocio.
  - `search_index_manager.py`: Utilidades para la gestión de índices de búsqueda.
  - `templates/`: Plantillas HTML para la interfaz web.
  - `static/`: Archivos estáticos servidos por el backend.
- **frontend/**: Aplicación frontend (React), interfaz de usuario para interactuar con el agente de IA.
  - `package.json`: Dependencias y scripts del frontend.
  - `src/`: Componentes React y lógica de UI.
- **data/**: Archivos de datos de ejemplo, embeddings, etc.
- **files/**: Archivos de ejemplo para pruebas de búsqueda y contexto.
- **Dockerfile**: Define cómo construir la imagen Docker para el backend.
- **requirements.txt / pyproject.toml**: Dependencias de Python para el backend.
- **logging_config.py**: Configuración de logging para la API.
- **gunicorn.conf.py**: Configuración del servidor Gunicorn para producción.

## Principales tecnologías usadas
- **Backend**: Python, FastAPI, Azure SDK, OpenAI
- **Frontend**: React, TypeScript, Fluent UI
- **Infraestructura**: Docker, Azure Container Apps, Azure AI Services

## Cómo ejecutar localmente

1. Instala las dependencias de Python:
   ```sh
   pip install -r requirements.txt
   ```
2. Instala las dependencias del frontend:
   ```sh
   cd frontend
   npm install
   # o pnpm install
   ```
3. Ejecuta el backend:
   ```sh
   uvicorn api.main:app --reload
   ```
4. Ejecuta el frontend:
   ```sh
   cd frontend
   npm start
   ```

## Notas
- Personaliza los componentes de React en `frontend/src/components` para modificar la UI.
- Modifica los endpoints en `api/routes.py` para cambiar la lógica del backend.
- Usa los archivos de ejemplo en `files/` y `data/` para pruebas y desarrollo.

---

**Autor:** Capgemini
