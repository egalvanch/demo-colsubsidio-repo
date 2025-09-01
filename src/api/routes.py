# Copyright (c) Microsoft.
# Licensed under the MIT license.

import asyncio
import json
import os
import hashlib
import logging
from typing import AsyncGenerator, Optional, Dict

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

    # Guardar solo si no es fallback
    fallback_text = "¡Uy! No encontré esa info"
    if cache_store_key and full_message and not full_message.strip().startswith(fallback_text):
        set_cached_response(cache_store_key, full_message)
        # También guardamos en cache global FAQ
        faq_key = faq_cache_key(request_str)
        set_cached_response(faq_key, full_message)

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

        faq_key = faq_cache_key(request_str)
        faq_cached = get_cached_response(faq_key)
        if faq_cached:
            logger.info("Respuesta enviada desde Cache Global.")
            response = StreamingResponse(string_streamer(faq_cached), headers=headers, media_type="text/event-stream")
            response.set_cookie("thread_id", thread_id)
            response.set_cookie("agent_id", agent_id)
            return response

        key = cache_key(thread_id, request_str)
        cached = get_cached_response(key)
        if cached:
            logger.info("Respuesta enviada desde Cache personal.")
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

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

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
