# Copyright (c) Microsoft.
# Licensed under the MIT license.

import asyncio
import json
import os
import hashlib
import logging
from typing import AsyncGenerator, Optional, Dict, List, Optional

import fastapi
from fastapi import Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry import trace

from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    Agent,
    MessageDeltaChunk,
    ThreadMessage,
    ThreadRun,
    AsyncAgentEventHandler,
    RunStep
)
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
   AgentEvaluationRequest,
   AgentEvaluationRedactionConfiguration,
   EvaluatorIds
)

import redis
from pydantic import BaseModel
import numpy as np
import hashlib
import httpx

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger("azureaiapp")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
tracer = trace.get_tracer(__name__)

# ------------------------------------------------------------------------------
# FastAPI router / templates
# ------------------------------------------------------------------------------
directory = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=directory)
router = fastapi.APIRouter()

# ------------------------------------------------------------------------------
# Auth (HTTP Basic)
# ------------------------------------------------------------------------------
security = HTTPBasic()
username = os.getenv("WEB_APP_USERNAME")
password = os.getenv("WEB_APP_PASSWORD")
basic_auth = bool(username and password)
threshold = float(os.getenv("AUTH_THRESHOLD", "0.65"))

def authenticate(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> None:
    if not basic_auth:
        logger.info("Skipping authentication: WEB_APP_USERNAME or WEB_APP_PASSWORD not set.")
        return
    import secrets
    correct_username = secrets.compare_digest(credentials.username, username)
    correct_password = secrets.compare_digest(credentials.password, password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return

auth_dependency = Depends(authenticate) if basic_auth else None

# ------------------------------------------------------------------------------
# App state helpers
# ------------------------------------------------------------------------------
def get_ai_project(request: Request) -> AIProjectClient:
    return request.app.state.ai_project

def get_agent_client(request: Request) -> AgentsClient:
    return request.app.state.agent_client

def get_agent(request: Request) -> Agent:
    return request.app.state.agent

def get_app_insights_conn_str(request: Request) -> Optional[str]:
    return getattr(request.app.state, "application_insights_connection_string", None)

# ------------------------------------------------------------------------------
# OpenAI / Azure OpenAI: client + helpers
# ------------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_KEY = str(os.getenv("AZURE_OPENAI_KEY"))
AZURE_EMBEDDINGS_ENDPOINT = str(os.getenv("AZURE_EMBEDDINGS_ENDPOINT"))

# Funciones auxiliares para caché semántico
async def get_embedding(text: str) -> Optional[List[float]]:
    """Genera embedding para el texto usando Azure OpenAI."""
    payload = {"input": text}
    headers = {
        "api-key": AZURE_OPENAI_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(AZURE_EMBEDDINGS_ENDPOINT, json=payload, headers=headers)
            data = response.json()
            return data["data"][0]["embedding"] if "data" in data else None
        except Exception as e:
            print(f"Error generando embedding: {e}")
            return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calcula similitud coseno entre dos vectores."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def find_similar_cached_response(text: str, redis_client, similarity_threshold: float = threshold) -> Optional[str]:
    """Busca respuestas cacheadas similares usando embeddings."""
    try:
        print(f"[DEBUG] Buscando similares para: '{text[:50]}...'")
        print(f"[DEBUG] Umbral de similitud: {similarity_threshold}")
        
        query_embedding = await get_embedding(text)
        if not query_embedding:
            print("[DEBUG] ❌ No se pudo generar embedding para la consulta")
            return None
        
        print(f"[DEBUG] ✅ Embedding generado, longitud: {len(query_embedding)}")
        
        # Obtener todas las claves del caché semántico
        cached_keys = redis_client.keys("semantic:*")
        
        print(f"[DEBUG] 📁 Claves encontradas en caché: {len(cached_keys)}")
        print(f"Cached Keys: {cached_keys}")
        
        best_similarity = 0
        best_response = None
        similarities = []
        
        for i, key in enumerate(cached_keys):
            cached_data = redis_client.get(key)
            if cached_data:
                cache_entry = json.loads(cached_data)
                cached_embedding = cache_entry.get("embedding")
                cached_text = cache_entry.get("text", "")[:50]
                
                if cached_embedding:
                    similarity = cosine_similarity(query_embedding, cached_embedding)
                    similarities.append((key, similarity, cached_text))
                    print(f"[DEBUG] 🔍 #{i+1} Similitud: {similarity:.4f} | Texto: '{cached_text}...'")
                    
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_response = cache_entry.get("response")
                        print(f"[DEBUG] 🎯 ¡Nueva mejor coincidencia! Similitud: {similarity:.4f}")
        
        # Mostrar ranking de similitudes
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] 📊 Top 3 similitudes:")
        for i, (key, sim, text) in enumerate(similarities[:3]):
            print(f"[DEBUG]   {i+1}. {sim:.4f} - '{text}...'")
        
        if best_response:
            print(f"[DEBUG] ✅ Respuesta encontrada con similitud: {best_similarity:.4f}")
        else:
            print(f"[DEBUG] ❌ No se encontró respuesta similar (mejor: {max([s[1] for s in similarities]) if similarities else 0:.4f})")
        
        return best_response
    except Exception as e:
        print(f"[DEBUG] 💥 Error buscando caché semántico: {e}")
        return None

async def cache_semantic_response(text: str, response: str) -> bool:
    """Guarda respuesta con embedding en caché semántico."""
    try:
        embedding = await get_embedding(text)
        if not embedding:
            return False
        
        cache_entry = {
            "text": text,
            "response": response,
            "embedding": embedding
        }
        
        # Usar hash del texto como parte de la clave
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        cache_key = f"semantic:{text_hash}"
        
        redis_client = get_redis_client()
        redis_client.set(cache_key, json.dumps(cache_entry), ex=3600)  # Expira en 1 hora
        print(f"Cache guardado en caché semántico")
        return True
    except Exception as e:
        print(f"Error guardando en caché semántico: {e}")
        return False

async def use_semantic_cache(text: str, similarity_threshold: float = threshold) -> Optional[str]:
    redis_client = get_redis_client()
    # Buscar respuesta similar en caché semántico
    cached_answer = await find_similar_cached_response(text, redis_client)
    print(f"[DEBUG] Respuesta cacheada: {cached_answer}, type: {type(cached_answer)}")
    if cached_answer:
        return cached_answer
    
    try:
        print(f"[DEBUG] Buscando similares para: '{text[:50]}...'")
        print(f"[DEBUG] Umbral de similitud: {similarity_threshold}")
        query_embedding = await get_embedding(text)
        if not query_embedding:
            print("[DEBUG] ❌ No se pudo generar embedding para la consulta")
            return None
        print(f"[DEBUG] ✅ Embedding generado, longitud: {len(query_embedding)}")
        
        # Obtener todas las claves del caché semántico
        cached_keys = redis_client.keys("semantic:*")
        
        print(f"[DEBUG] 📁 Claves encontradas en caché: {len(cached_keys)}")
        print(f"Cached Keys: {cached_keys}")
        
        best_similarity = 0
        best_response = None
        similarities = []
        
        for i, key in enumerate(cached_keys):
            cached_data = redis_client.get(key)
            if cached_data:
                cache_entry = json.loads(cached_data)
                cached_embedding = cache_entry.get("embedding")
                cached_text = cache_entry.get("text", "")[:50]
                
                if cached_embedding:
                    similarity = cosine_similarity(query_embedding, cached_embedding)
                    similarities.append((key, similarity, cached_text))
                    print(f"[DEBUG] 🔍 #{i+1} Similitud: {similarity:.4f} | Texto: '{cached_text}...'")
                    
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_response = cache_entry.get("response")
                        print(f"[DEBUG] 🎯 ¡Nueva mejor coincidencia! Similitud: {similarity:.4f}")
        
        # Mostrar ranking de similitudes
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] 📊 Top 3 similitudes:")
        for i, (key, sim, text) in enumerate(similarities[:3]):
            print(f"[DEBUG]   {i+1}. {sim:.4f} - '{text}...'")
        
        if best_response:
            print(f"[DEBUG] ✅ Respuesta encontrada con similitud: {best_similarity:.4f}")
        else:
            print(f"[DEBUG] ❌ No se encontró respuesta similar (mejor: {max([s[1] for s in similarities]) if similarities else 0:.4f})")
        
        return best_response
    except Exception as e:
        print(f"[DEBUG] 💥 Error buscando caché semántico: {e}")
        return None
    

# ------------------------------------------------------------------------------
# Redis: singleton client + helpers
# ------------------------------------------------------------------------------
_redis_client: Optional[redis.StrictRedis] = None
_CACHE_ENABLED = False
_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "86400"))  # 24h por defecto

def _init_redis_once() -> None:
    global _redis_client, _CACHE_ENABLED
    if _redis_client is not None:
        return
    host = os.getenv("REDIS_HOST")
    pwd = os.getenv("REDIS_PASSWORD")
    port = os.getenv("REDIS_PORT")
    if not (host and pwd and port):
        logger.warning("Redis not fully configured (REDIS_HOST/PORT/PASSWORD). Cache disabled.")
        _CACHE_ENABLED = False
        _redis_client = None
        return
    try:
        _redis_client = redis.StrictRedis(
            host=str(host),
            port=int(port),
            password=str(pwd),
            ssl=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            retry_on_timeout=True
        )
        try:
            _redis_client.ping()
            _CACHE_ENABLED = True
            logger.info("Redis cache enabled.")
        except Exception as ping_err:
            logger.warning(f"Redis ping failed. Cache disabled. Err: {ping_err}")
            _CACHE_ENABLED = False
            _redis_client = None
    except Exception as e:
        logger.error(f"Error creating Redis client. Cache disabled. Err: {e}")
        _CACHE_ENABLED = False
        _redis_client = None

def get_redis_client() -> Optional[redis.StrictRedis]:
    if _redis_client is None:
        _init_redis_once()
    return _redis_client if _CACHE_ENABLED else None

# ------------------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------------------
def normalize_message(text: str) -> str:
    return (text or "").strip().lower()

def cache_key(thread_id: str, user_message: str) -> str:
    normalized = normalize_message(user_message)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"thread:{thread_id}:msg:{digest}"

def faq_cache_key(user_message: str) -> str:
    normalized = normalize_message(user_message)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"faq:msg:{digest}"

def get_cached_response(key: str) -> Optional[str]:
    client = get_redis_client()
    if not client:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
    except Exception as e:
        logger.warning(f"Redis GET failed for key={key}: {e}")
        return None

def set_cached_response(key: str, content: str, ttl: int = _CACHE_TTL) -> None:
    client = get_redis_client()
    if not client:
        return
    try:
        client.set(key, content, ex=ttl)
    except Exception as e:
        logger.warning(f"Redis SET failed for key={key}: {e}")

# ------------------------------------------------------------------------------
# FAQs iniciales en cache
# ------------------------------------------------------------------------------
INITIAL_FAQS: Dict[str, str] = {}

def preload_faqs():
    client = get_redis_client()
    if not client:
        logger.warning("Redis no disponible, no se cargaron FAQs iniciales.")
        return
    for question, answer in INITIAL_FAQS.items():
        key = faq_cache_key(question)
        if not client.exists(key):
            client.set(key, answer, ex=_CACHE_TTL)
            logger.info(f"FAQ precargada en cache: {question}")

# ------------------------------------------------------------------------------
# SSE helper
# ------------------------------------------------------------------------------
def serialize_sse_event(data: Dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

def string_streamer(text: str):
    data = {
        "type": "completed_message",
        "role": "assistant",
        "content": text
    }
    yield f"data: {json.dumps(data)}\n\n"

# ------------------------------------------------------------------------------
# Agent message/annotation helpers
# ------------------------------------------------------------------------------
async def get_message_and_annotations(agent_client: AgentsClient, message: ThreadMessage) -> Dict:
    annotations = []
    for ann in (a.as_dict() for a in message.file_citation_annotations):
        file_id = ann["file_citation"]["file_id"]
        try:
            openai_file = await agent_client.files.get(file_id)
            ann["file_name"] = openai_file.filename
        except Exception as e:
            logger.warning(f"Could not fetch file name for file_id={file_id}: {e}")
        annotations.append(ann)
    for url_ann in message.url_citation_annotations:
        ann = url_ann.as_dict()
        ann["file_name"] = ann["url_citation"]["title"]
        annotations.append(ann)
    return {
        "content": message.text_messages[0].text.value if message.text_messages else "",
        "annotations": annotations
    }

class MyEventHandler(AsyncAgentEventHandler[str]):
    def __init__(self, ai_project: AIProjectClient, app_insights_conn_str: Optional[str]):
        super().__init__()
        self.agent_client = ai_project.agents
        self.ai_project = ai_project
        self.app_insights_conn_str = app_insights_conn_str or ""

    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        return serialize_sse_event({"content": delta.text, "type": "message"})

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        if message.status != "completed":
            return None
        stream_data = await get_message_and_annotations(self.agent_client, message)
        stream_data["type"] = "completed_message"
        return serialize_sse_event(stream_data)

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        stream_data = {"content": f"ThreadRun status: {run.status}, thread ID: {run.thread_id}", "type": "thread_run"}
        if run.status == "failed":
            stream_data["error"] = str(run.last_error)
        if run.status == "completed":
            run_agent_evaluation(run.thread_id, run.id, self.ai_project, self.app_insights_conn_str)
        return serialize_sse_event(stream_data)

    async def on_error(self, data: str) -> Optional[str]:
        logger.error(f"MyEventHandler: on_error: {data}")
        return serialize_sse_event({"type": "stream_end"})

    async def on_done(self) -> Optional[str]:
        return serialize_sse_event({"type": "stream_end"})

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@router.on_event("startup")
async def startup_event():
    preload_faqs()

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, _ = auth_dependency):
    return templates.TemplateResponse("index.html", {"request": request})

async def get_result(
    request_str: str,
    request: Request,
    thread_id: str,
    agent_id: str,
    ai_project: AIProjectClient,
    app_insight_conn_str: Optional[str],
    carrier: Dict[str, str],
    cache_store_key: Optional[str] = None
) -> AsyncGenerator[str, None]:
    ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    full_message = ""

    with tracer.start_as_current_span("get_result", context=ctx):
        try:
            agent_client = ai_project.agents
            stream_cm = await agent_client.runs.stream(
                thread_id=thread_id,
                agent_id=agent_id,
                event_handler=MyEventHandler(ai_project, app_insight_conn_str),
            )
            async with stream_cm as stream:
                yield serialize_sse_event({"type": "info", "message": "stream started"})
                async for event in stream:
                    _, _, event_func_return_val = event
                    if not event_func_return_val:
                        continue
                    try:
                        data = json.loads(event_func_return_val.replace("data: ", "").strip())
                        if data.get("type") == "message" and "content" in data:
                            full_message += data["content"]
                    except Exception:
                        pass
                    yield event_func_return_val
        except Exception as e:
            logger.exception(f"get_result: Exception: {e}")
            yield serialize_sse_event({"type": "error", "message": str(e)})


    errores = ["No se ha encontrado", "No he podido encontrar", "No puedo ayudarte con eso"]
    es_fallback = any(err in full_message for err in errores)
    if cache_store_key and full_message and not es_fallback:
        await cache_semantic_response(request_str, full_message)
        set_cached_response(cache_store_key, full_message)
        # También guardamos en cache global FAQ
        faq_key = faq_cache_key(request_str)
        #set_cached_response(faq_key, full_message)
        save_thread_interaction(thread_id, request_str, full_message)

@router.get("/chat/history")
async def history(
    request: Request,
    ai_project : AIProjectClient = Depends(get_ai_project),
    agent : Agent = Depends(get_agent),
	_ = auth_dependency
):
    with tracer.start_as_current_span("chat_history"):
        # Retrieve the thread ID from the cookies (if available).
        thread_id = request.cookies.get('thread_id')
        agent_id = request.cookies.get('agent_id')

        # Attempt to get an existing thread. If not found, create a new one.
        try:
            agent_client = ai_project.agents
            if thread_id and agent_id == agent.id:
                logger.info(f"Retrieving thread with ID {thread_id}")
                thread = await agent_client.threads.get(thread_id)
            else:
                logger.info("Creating a new thread")
                thread = await agent_client.threads.create()
        except Exception as e:
            logger.error(f"Error handling thread: {e}")
            raise HTTPException(status_code=400, detail=f"Error handling thread: {e}")

        thread_id = thread.id
        agent_id = agent.id

    # Create a new message from the user's input.
    try:
        content = []
        response = agent_client.messages.list(
            thread_id=thread_id,
        )
        async for message in response:
            formatteded_message = await get_message_and_annotations(agent_client, message)
            formatteded_message['role'] = message.role
            formatteded_message['created_at'] = message.created_at.astimezone().strftime("%m/%d/%y, %I:%M %p")
            content.append(formatteded_message)


        logger.info(f"List message, thread ID: {thread_id}")
        response = JSONResponse(content=content)

        # Update cookies to persist the thread and agent IDs.
        response.set_cookie("thread_id", thread_id)
        response.set_cookie("agent_id", agent_id)
        return response
    except Exception as e:
        logger.error(f"Error listing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error list message: {e}")

@router.get("/agent")
async def get_chat_agent(request: Request):
    return JSONResponse(content=get_agent(request).as_dict())

@router.post("/chat")
async def chat(
    request: Request,
    agent: Agent = Depends(get_agent),
    ai_project: AIProjectClient = Depends(get_ai_project),
    app_insights_conn_str: Optional[str] = Depends(get_app_insights_conn_str),
    _ = auth_dependency
):
    thread_id = request.cookies.get("thread_id")
    agent_id = request.cookies.get("agent_id")

    with tracer.start_as_current_span("chat_request"):
        carrier: Dict[str, str] = {}
        TraceContextTextMapPropagator().inject(carrier)

        try:
            agent_client = ai_project.agents
            if thread_id and agent_id == agent.id:
                thread = await agent_client.threads.get(thread_id)
            else:
                thread = await agent_client.threads.create()
        except Exception as e:
            logger.error(f"Error handling thread: {e}")
            raise HTTPException(status_code=400, detail=f"Error handling thread: {e}")

        thread_id = thread.id
        agent_id = agent.id

        try:
            user_payload = await request.json()
            request_str: str = (user_payload.get("message") or "").strip()
        except Exception as e:
            logger.error(f"Invalid JSON in request: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in request: {e}")

        if not request_str:
            raise HTTPException(status_code=400, detail="Empty 'message' is not allowed.")

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
        }
        key = cache_key(thread_id, request_str)
        cached = get_cached_response(key)
        
        semantic_cache = await use_semantic_cache(request_str)
        print(f"[DEBUG] Respuesta cacheada semántica: {semantic_cache}, type: {type(semantic_cache)}")
        if (semantic_cache):
            logger.info("Respuesta enviada desde Cache Semántico.")
            set_cached_response(key, semantic_cache)
            response = StreamingResponse(string_streamer(semantic_cache), headers=headers, media_type="text/event-stream")
            response.set_cookie("thread_id", thread_id)
            response.set_cookie("agent_id", agent_id)
            return response

        faq_key = faq_cache_key(request_str)
        faq_cached = get_cached_response(faq_key)

        if faq_cached:
            logger.info("Respuesta enviada desde Cache Global.")
            await cache_semantic_response(request_str, faq_cached)
            response = StreamingResponse(string_streamer(faq_cached), headers=headers, media_type="text/event-stream")
            response.set_cookie("thread_id", thread_id)
            response.set_cookie("agent_id", agent_id)
            return response


        if cached:
            logger.info("Respuesta enviada desde Cache personal.")
            await cache_semantic_response(request_str, cached)
            response = StreamingResponse(string_streamer(cached), headers=headers, media_type="text/event-stream")
        else:
            try:
                await agent_client.messages.create(thread_id=thread_id, role="user", content=request_str)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error creating message: {e}")
            logger.info("Respuesta generada desde agente.")
            response = StreamingResponse(
                get_result(request_str, request, thread_id, agent_id, ai_project, app_insights_conn_str, carrier, key),
                headers=headers,
                media_type="text/event-stream"
            )

        response.set_cookie("thread_id", thread_id)
        response.set_cookie("agent_id", agent_id)
        return response
@router.get("/config/azure")
async def get_azure_config(_ = auth_dependency):
    """Get Azure configuration for frontend use"""
    try:
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        tenant_id = os.environ.get("AZURE_TENANT_ID", "")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "")
        ai_project_resource_id = os.environ.get("AZURE_EXISTING_AIPROJECT_RESOURCE_ID", "")
        
        # Extract resource name and project name from the resource ID
        # Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices/accounts/{resource}/projects/{project}
        resource_name = ""
        project_name = ""
        
        if ai_project_resource_id:
            parts = ai_project_resource_id.split("/")
            if len(parts) >= 8:
                resource_name = parts[8]  # accounts/{resource_name}
            if len(parts) >= 10:
                project_name = parts[10]  # projects/{project_name}
        
        return JSONResponse({
            "subscriptionId": subscription_id,
            "tenantId": tenant_id,
            "resourceGroup": resource_group,
            "resourceName": resource_name,
            "projectName": project_name,
            "wsid": ai_project_resource_id
        })
    except Exception as e:
        logger.error(f"Error getting Azure config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Azure configuration")

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
    
def save_thread_interaction(thread_id: str, user_message: str, content: str, role: str = "visitante") -> None:
    client = get_redis_client()
    if not client:
        return
    try:
        key = f"thread:{thread_id}:rol:{role}"
        existing = client.get(key)
        interactions = []
        if existing:
            try:
                interactions = json.loads(existing)
            except Exception:
                interactions = []
        interactions.append({"user_message": user_message, "content": content})
        client.set(key, json.dumps(interactions), ex=_CACHE_TTL)
    except Exception as e:
        logger.warning(f"Redis append interaction failed for key={key}: {e}")

def run_agent_evaluation(
    thread_id: str,
    run_id: str,
    ai_project: AIProjectClient,
    app_insights_conn_str: Optional[str]
):
    if not app_insights_conn_str:
        return
    agent_evaluation_request = AgentEvaluationRequest(
        run_id=run_id,
        thread_id=thread_id,
        evaluators={
            "Relevance": {"Id": EvaluatorIds.RELEVANCE.value},
            "TaskAdherence": {"Id": EvaluatorIds.TASK_ADHERENCE.value},
            "ToolCallAccuracy": {"Id": EvaluatorIds.TOOL_CALL_ACCURACY.value},
        },
        sampling_configuration=None,
        redaction_configuration=AgentEvaluationRedactionConfiguration(
            redact_score_properties=False,
        ),
        app_insights_connection_string=app_insights_conn_str,
    )
    def _run():
        try:
            ai_project.evaluations.create_agent_evaluation(evaluation=agent_evaluation_request)
        except Exception as e:
            logger.error(f"Error creating agent evaluation: {e}")
