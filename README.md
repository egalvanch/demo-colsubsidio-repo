# Asistente Virtual con IA Generativa – Colsubsidio

**Propósito.** Diseñar e implementar un **asistente virtual** para **empresas Pyme/Micro** y para el **equipo comercial** de Colsubsidio, que agilice afiliaciones, resuelva dudas de servicios/beneficios y eleve la satisfacción de usuarios, apoyado en Azure AI Foundry, evaluaciones y trazabilidad end‑to‑end. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

> Este README resume alcance, arquitectura, canales, plan de trabajo, indicadores y factores críticos de éxito de acuerdo con la planeación validada con Colsubsidio. Fuentes de referencia: **[Ppt Asistente Virtual.pdf](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf?EntityRepresentationId=a433ec6c-66d1-4cd4-85c0-a89861ace445)** y **[OT - 20250808_V1.0.pdf](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf?EntityRepresentationId=9a17f074-ed48-473f-9db7-68b03cd1b9a8)**. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)[1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

---

## 1) Contexto y Objetivos

**Contexto actual.** La atención a Pymes/Micro es fragmentada; hay alta demanda de información (Portal Empresas/Portal en Línea), baja comunicación del portafolio y comprensión limitada de necesidades, generando experiencias insatisfactorias y riesgo de fuga. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)

**Objetivo general.** Un asistente virtual que permita a empresas y empleados gestionar afiliaciones, resolver consultas y facilitar acceso a beneficios, optimizando la experiencia. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

**Objetivos específicos.**
- Mejorar la eficiencia del proceso de **afiliación**. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)  
- Incrementar el **uso de beneficios** disponibles. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)  
- Resolver dudas sobre **servicios** y orientar en su aprovechamiento. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)  
- **Fortalecer la atención** a empresas sin ejecutivo o con sobrecarga. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

---

## 2) Alcance por Fases

- **Fase 0 – Evaluación y Descubrimiento.** Aterrizar segmentos, journeys y fuentes; definir KPIs y criterios de éxito. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)
- **Fase 1 – 2 productos (MVP informativo/consultivo).**  
  **Producto 1:** Asistente para Pymes/Micro sin ejecutivo.  
  **Producto 2:** Asistente interno para el equipo comercial. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)
- **Fases II–IV (Roadmap).** Evolución transaccional por línea de negocio, mantenimiento de contexto omnicanal, escalabilidad técnica y gobernanza de datos. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

---

## 3) Usuarios y Canales

**Usuarios:**  
1) Representante de empresa no afiliada. 2) Representante de empresa afiliada.  
3) Usuario de empresa no afiliada. 4) Usuario afiliado. 5) Ejecutivo de cuenta. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

**Canales prioritarios:** **Web/Portal**, **WhatsApp**, **IVR** y **Microsoft Teams** (para ejecutivos). [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

---

## 4) Arquitectura de Solución (resumen)

- **Frontend**: Web App (chat) para usuarios empresariales.  
- **Backend/Agente**: Azure Container Apps ejecuta el servicio que orquesta el **Azure AI Agent** y herramientas (file‑search / conocimiento).  
- **Modelo**: despliegue base de `gpt‑4o‑mini` y embeddings (`text-embedding-3-small`), ajustables por entorno.  
- **Azure AI Foundry**: proyecto con evaluación del agente y trazabilidad.  
- **Almacenamiento**: Azure Storage para archivos y estados.  
- **Observabilidad**: Application Insights + Log Analytics para métricas, trazas y logs.  
- **Opcional**: Azure AI Search para RAG híbrido (vectorial + semántico) según fuentes validadas.

> La arquitectura está alineada al enfoque de “asistente informativo/consultivo” con monitoreo y trazabilidad integrados, según las láminas y entregables de planeación. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)[1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

---

## 5) Funcionalidades Clave del MVP (Fase 1)

- **Búsqueda y presentación de información** de afiliaciones, beneficios y procesos, con respuestas citadas cuando aplique. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)  
- **Soporte al ejecutivo comercial** con consulta rápida de portafolio/procesos y guías operativas. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)  
- **Omnicanal** inicial (Web, WhatsApp, IVR, Teams) y **desborde a humano** cuando sea necesario. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)  
- **Trazabilidad y evaluación** del agente para mejorar calidad y reducir tiempos de atención. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

---

## 6) Indicadores de Éxito

- **Índice de fuga** de Pymes/Micro. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)  
- **Cobertura de usabilidad** de servicios B2B y **contactabilidad**. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)  
- **NPS** por segmento (Empresarial TOP/Estándar/Micro). [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)  
- **Presupuesto de venta** de servicios (impacto comercial). [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)

> En la comunicación de valor se proyecta un fuerte impacto en volumen de consultas, tiempos de respuesta y finalización de solicitudes de beneficios, como base de narrativa para adopción y ROI. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

---

## 7) Plan de Trabajo (alto nivel)

- **Duración de referencia**: 16 semanas para el plan completo (con releases M1–M4).  
- **Entregables MVP informativo**: bot funcional, interfaz web, reportes de uso, integración con canales; onboarding interno para fuerza comercial.  
- **Integraciones** (roadmap): SAP/HANA Cloud, desborde a chat asistido y personalización por perfil. [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)
### Cronograma sugerido
- **M1 – Diseño y planeación** (descubrimiento, datos, KPIs, criterios de calidad). [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)  
- **M2 – Desarrollo informativo (MVP)** (contenido, prompts, políticas, canal web). [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)  
- **M3 – QA, liberación e instrumentación** (trazas, métricas, dashboards). [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)  
- **M4 – Capacitación y habilitación comercial** (playbooks, sesiones, feedback). [1](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/OT%20-%2020250808_V1.0.pdf)

---

## 8) Factores Críticos de Éxito

- Custodiar **alcance y objetivos** de esta fase.  
- **Disponibilidad** de equipos involucrados.  
- Asegurar **estructura y roles** para adopción del modelo operativo.  
- **Comunicación** sostenida y **monitoreo continuo** de riesgos.  
- **Prioridad ejecutiva** y espacio de trabajo para tracción. [2](https://capgemini-my.sharepoint.com/personal/emiliano_galvan_capgemini_com/Documents/Microsoft%20Copilot%20Chat%20Files/Ppt%20Asistente%20Virtual.pdf)

---

## 9) Despliegue (rápido) – Desarrollo / Demo

> Requisitos: `az` CLI, suscripción con permisos para crear recursos, acceso a Azure AI Foundry y Container Apps.

```bash
# 1) Autenticación
az login

# 2) Selecciona suscripción
az account set --subscription "<SUBSCRIPTION_ID>"

# 3) Provisiona infraestructura base (grupo de recursos, Foundry, Container Apps, Storage, AppInsights, LogAnalytics)
# (usa tus plantillas/azd; si empleas 'azd', asegúrate de tener .env parametrizado)
azd up
