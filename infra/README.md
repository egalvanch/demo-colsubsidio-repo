# Infraestructura para Azure AI Agents

Esta carpeta contiene la definición de la infraestructura como código (IaC) necesaria para desplegar y operar la solución de agentes de IA en Azure.

## Estructura

- **main.bicep**: Archivo principal que orquesta la creación de todos los recursos de Azure requeridos.
- **main.parameters.json**: Parámetros para personalizar el despliegue (pueden ser variables de entorno o valores fijos).
- **core/**: Subcarpetas con módulos Bicep reutilizables para recursos específicos:
  - **ai/**: Recursos de Azure AI (Cognitive Services, AI Projects, etc.)
  - **host/**: Recursos de cómputo (Container Apps, App Service, etc.)
  - **monitor/**: Monitorización y logging (Log Analytics, Application Insights).
  - **search/**: Azure Cognitive Search y conexiones.
  - **security/**: Key Vault, identidades y roles.
  - **storage/**: Cuentas de almacenamiento y conexiones.
- **abbreviations.json**: Abreviaciones para nombres de recursos.

## Recursos que se despliegan
- Azure Resource Group
- Azure AI Services (OpenAI, Cognitive Services)
- Azure AI Project
- Azure Cognitive Search (opcional)
- Azure Storage Account
- Azure Container Registry
- Azure Log Analytics Workspace
- Azure Application Insights (opcional)
- Azure Key Vault
- Identidades administradas y asignaciones de roles

## Personalización
- Puedes definir nombres fijos para los recursos editando los parámetros en `main.parameters.json` o pasándolos al momento del despliegue.
- Variables como `principalId` pueden obtenerse automáticamente de recursos con identidad administrada.
- Habilita o deshabilita servicios opcionales (Search, App Insights) mediante parámetros.

## Despliegue

### Manual
1. Asegúrate de tener [Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli) y permisos suficientes.
2. Autentícate: `az login`
3. Despliega:
   ```sh
   az deployment sub create --location <region> --template-file ./main.bicep --parameters @main.parameters.json
   ```
   O usando parámetros individuales:
   ```sh
   az deployment sub create --location <region> --template-file ./main.bicep -p environmentName=dev -p aiServicesName=mitiendaaiservice ...
   ```

### Con Azure Developer CLI (azd)
1. Instala [azd](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd)
2. Ejecuta:
   ```sh
   azd up
   ```

### En CI/CD (Azure DevOps, GitHub Actions)
- Define las variables de entorno requeridas en el pipeline.
- El pipeline usará los archivos de esta carpeta para desplegar la infraestructura automáticamente.

## Notas
- Modifica los archivos Bicep para personalizar la arquitectura según tus necesidades.
- Consulta la documentación oficial de Azure para detalles sobre cada recurso.
- Elimina los recursos con `azd down` o el comando correspondiente para evitar costos innecesarios.

---

**Autor:** Capgemini
