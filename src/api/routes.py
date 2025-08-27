# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import asyncio
import json
import os
from typing import AsyncGenerator, Optional, Dict
import io

import fastapi
from fastapi import Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

import logging
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

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
   AgentEvaluationSamplingConfiguration,
   AgentEvaluationRedactionConfiguration,
   EvaluatorIds
)

# Create a logger for this module
logger = logging.getLogger("azureaiapp")

# Set the log level for the azure HTTP logging policy to WARNING (or ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

from opentelemetry import trace
tracer = trace.get_tracer(__name__)

# Define the directory for your templates.
directory = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=directory)

# Create a new FastAPI router
router = fastapi.APIRouter()

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
import secrets

security = HTTPBasic()

username = os.getenv("WEB_APP_USERNAME")
password = os.getenv("WEB_APP_PASSWORD")
basic_auth = username and password

def authenticate(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> None:

    if not basic_auth:
        logger.info("Skipping authentication: WEB_APP_USERNAME or WEB_APP_PASSWORD not set.")
        return
    
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

# Redis connection
import redis
def connect_redis():
    myHostname = str(os.getenv("REDIS_HOST"))
    myPassword = str(os.getenv("REDIS_PASSWORD"))
    myPort = str(os.getenv("REDIS_PORT"))
    return redis.StrictRedis(host=myHostname, port=int(myPort), password=myPassword, ssl=True)

def message_in_cache(message: str) -> bool:
    return connect_redis().exists(message)
    
def set_redis_cache(message: str, response: str) -> None:
    return connect_redis().set(message, response, ex=3600)

def get_ai_project(request: Request) -> AIProjectClient:
    return request.app.state.ai_project

def get_agent_client(request: Request) -> AgentsClient:
    return request.app.state.agent_client

def get_agent(request: Request) -> Agent:
    return request.app.state.agent

def get_app_insights_conn_str(request: Request) -> str:
    if hasattr(request.app.state, "application_insights_connection_string"):
        return request.app.state.application_insights_connection_string
    else:
        return None

def serialize_sse_event(data: Dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

async def get_message_and_annotations(agent_client : AgentsClient, message: ThreadMessage) -> Dict:
    annotations = []
    # Get file annotations for the file search.
    for annotation in (a.as_dict() for a in message.file_citation_annotations):
        file_id = annotation["file_citation"]["file_id"]
        logger.info(f"Fetching file with ID for annotation {file_id}")
        openai_file = await agent_client.files.get(file_id)
        annotation["file_name"] = openai_file.filename
        logger.info(f"File name for annotation: {annotation['file_name']}")
        annotations.append(annotation)

    # Get url annotation for the index search.
    for url_annotation in message.url_citation_annotations:
        annotation = url_annotation.as_dict()
        annotation["file_name"] = annotation['url_citation']['title']
        logger.info(f"File name for annotation: {annotation['file_name']}")
        annotations.append(annotation)
            
    return {
        'content': message.text_messages[0].text.value,
        'annotations': annotations
    }

class MyEventHandler(AsyncAgentEventHandler[str]):
    def __init__(self, ai_project: AIProjectClient, app_insights_conn_str: Optional[str]):
        super().__init__()
        self.agent_client = ai_project.agents
        self.ai_project = ai_project
        self.app_insights_conn_str = app_insights_conn_str or ""

    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        logger.info(f"on_message_delta: delta.text={delta.text}")
        stream_data = {'content': delta.text, 'type': "message"}
        logger.info(f"on_message_delta: stream_data={stream_data}")
        return serialize_sse_event(stream_data)

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        try:
            logger.info(f"on_thread_message: Received thread message, message ID: {message.id}, status: {message.status}")
            if message.status != "completed":
                logger.info("on_thread_message: message not completed, returning None")
                return None

            logger.info("on_thread_message: Received completed message, fetching annotations")
            stream_data = await get_message_and_annotations(self.agent_client, message)
            logger.info(f"on_thread_message: stream_data={stream_data}")
            stream_data['type'] = "completed_message"
            logger.info(f"on_thread_message: returning completed_message event")
            return serialize_sse_event(stream_data)
        except Exception as e:
            logger.error(f"Error in event handler for thread message: {e}", exc_info=True)
            return None

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        logger.info(f"on_thread_run: event received, status={run.status}, thread_id={run.thread_id}")
        run_information = f"ThreadRun status: {run.status}, thread ID: {run.thread_id}"
        stream_data = {'content': run_information, 'type': 'thread_run'}
        if run.status == "failed":
            logger.info(f"on_thread_run: run failed, error={run.last_error}")
            stream_data['error'] = str(run.last_error)
        if run.status == "completed":
            logger.info("on_thread_run: run completed, starting agent evaluation")
            run_agent_evaluation(run.thread_id, run.id, self.ai_project, self.app_insights_conn_str)
        logger.info(f"on_thread_run: returning thread_run event")
        return serialize_sse_event(stream_data)

    async def on_error(self, data: str) -> Optional[str]:
        logger.error(f"MyEventHandler: on_error event received: {data}")
        stream_data = {'type': "stream_end"}
        return serialize_sse_event(stream_data)

    async def on_done(self) -> Optional[str]:
        logger.info("MyEventHandler: on_done event received")
        stream_data = {'type': "stream_end"}
        return serialize_sse_event(stream_data)

    async def on_run_step(self, step: RunStep) -> Optional[str]:
        logger.info(f"Step {step['id']} status: {step['status']}")
        step_details = step.get("step_details", {})
        tool_calls = step_details.get("tool_calls", [])

        if tool_calls:
            logger.info("Tool calls:")
            for call in tool_calls:
                azure_ai_search_details = call.get("azure_ai_search", {})
                if azure_ai_search_details:
                    logger.info(f"azure_ai_search input: {azure_ai_search_details.get('input')}")
                    logger.info(f"azure_ai_search output: {azure_ai_search_details.get('output')}")
        return None

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, _ = auth_dependency):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
        }
    )


async def get_result(
    request_str: str,
    request: Request, 
    thread_id: str, 
    agent_id: str, 
    ai_project: AIProjectClient,
    app_insight_conn_str: Optional[str], 
    carrier: Dict[str, str]
) -> AsyncGenerator[str, None]:
    ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    logger.info(f"get_result: extracted context, thread_id={thread_id}, agent_id={agent_id}")
    full_message = ""  # <-- Acumulador para el mensaje generado
    
    with tracer.start_as_current_span('get_result', context=ctx):
        logger.info(f"get_result: span started for thread_id={thread_id} and agent_id={agent_id}")
        try:
            agent_client = ai_project.agents
            logger.info("get_result: about to await agent_client.runs.stream")
            stream_cm = await agent_client.runs.stream(
                thread_id=thread_id, 
                agent_id=agent_id,
                event_handler=MyEventHandler(ai_project, app_insight_conn_str),
            )
            async with stream_cm as stream:
                logger.info("get_result: Successfully created stream; starting to process events")
                yield serialize_sse_event({'type': 'info', 'message': 'stream started'})
                logger.info("get_result: yielded stream started event")
                async for event in stream:
                    logger.info(f"get_result: received event from stream: {event}")
                    _, _, event_func_return_val = event
                    if event_func_return_val:
                        # Extrae el texto del chunk y acumúlalo
                        try:
                            data = json.loads(event_func_return_val.replace("data: ", "").strip())
                            if data.get("type") == "message" and "content" in data:
                                full_message += data["content"]
                        except Exception as e:
                            logger.warning(f"No se pudo parsear chunk SSE: {e}")
                        yield event_func_return_val
                    else:
                        logger.info("get_result: event received but no data to yield")
                logger.info("get_result: finished processing all events from stream")
        except Exception as e:
            logger.exception(f"get_result: Exception in get_result: {e}")
            yield serialize_sse_event({'type': "error", 'message': str(e)})
    # Al finalizar el stream, guarda el mensaje completo en Redis
    if full_message:
        #print(f"full_message to cache: {full_message}")
        redis = connect_redis()
        original_from_cache = redis.get(f"thread:{thread_id}:rol:visitante")
        if original_from_cache:
            message_for_cache = json.loads(original_from_cache)
            message_for_cache.append(
                {"user_message": request_str, "content": full_message}
            )
        else:
            message_for_cache = [
                {"user_message": request_str, "content": full_message}
            ]
        #print(f"message_for_cache to set in redis: {message_for_cache}")
        redis.set(f"thread:{thread_id}:rol:visitante", json.dumps(message_for_cache), ex=3600)


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
async def get_chat_agent(
    request: Request
):
    return JSONResponse(content=get_agent(request).as_dict())  

@router.post("/chat")
async def chat(
    request: Request,
    agent : Agent = Depends(get_agent),
    ai_project: AIProjectClient = Depends(get_ai_project),
    app_insights_conn_str : str = Depends(get_app_insights_conn_str),
	_ = auth_dependency
):
    # Retrieve the thread ID from the cookies (if available).
    thread_id = request.cookies.get('thread_id')
    agent_id = request.cookies.get('agent_id')
    
    with tracer.start_as_current_span("chat_request"):
        carrier = {}        
        TraceContextTextMapPropagator().inject(carrier)
        
        # Attempt to get an existing thread. If not found, create a new one.
        try:
            agent_client = ai_project.agents
            logger.info("Preparando para obtener o crear thread...")
            if thread_id and agent_id == agent.id:
                logger.info(f"Retrieving thread with ID {thread_id}")
                thread = await agent_client.threads.get(thread_id)
                logger.info(f"Thread obtenido: {thread}")
            else:
                logger.info("Creating a new thread")
                thread = await agent_client.threads.create()
                logger.info(f"Thread creado: {thread}")
        except Exception as e:
            logger.error(f"Error handling thread: {e}")
            raise HTTPException(status_code=400, detail=f"Error handling thread: {e}")

        thread_id = thread.id
        agent_id = agent.id

        # Parse the JSON from the request.
        try:
            user_message = await request.json()
            request_str = user_message.get('message', '')
        except Exception as e:
            logger.error(f"Invalid JSON in request: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in request: {e}")

        logger.info(f"user_message: {user_message}")
        
        # Connect to Redis
        logger.info("Intentando conectar a Redis...")
        try:
            redis = connect_redis()
            logger.info("Conexión a Redis exitosa")
        except Exception as e:
            logger.error(f"Error al conectar a Redis: {e}")
            raise HTTPException(status_code=500, detail=f"Error al conectar a Redis: {e}")
        
        # Recupera el último mensaje del usuario desde Redis (si existe)
        logger.info(f"Buscando último mensaje en Redis para thread:{thread_id}")
        try:
            message_from_cache = redis.get(f"thread:{thread_id}:rol:visitante")
            #print(f"message_from_cache obtenido de redis: {message_from_cache}; type: {type(message_from_cache)}")
            message_from_cache_object = json.loads(message_from_cache)
            #print(f"message_from_cache_object parseado: {message_from_cache_object}; type: {type(message_from_cache_object)}")
            
            for message in message_from_cache_object:
                if message['user_message'].strip().lower() == request_str.strip().lower():
                    print(f"El mensaje en cache coincide con el mensaje actual del usuario: {message['user_message']} == {request_str}")
                    message_from_cache = message
                    message_content = message['content']
                    break
                else:
                    message_from_cache = None
                    #logger.info("El mensaje en cache no coincide con el mensaje actual del usuario.")
        except Exception as e:
            logger.error(f"Error al leer de Redis: {e}")
            message_from_cache = None
        # message = None
        # if message_from_cache:
        #     try:
        #         # Si es bytes, decodifica; si es str, usa directo
        #         if isinstance(message_from_cache, bytes):
        #             message_str = message_from_cache.decode("utf-8")
        #         else:
        #             message_str = message_from_cache
        #         message = json.loads(message_str)
        #         logger.info(f"Último mensaje del usuario en cache: {message}")
        #     except Exception as e:
        #         logger.error(f"Error decodificando mensaje de Redis: {e}")
        #         message = None
        # if not message:
        #     # Create a new message from the user's input.
        #     try:
        #         logger.info("Creando nuevo mensaje de usuario en el thread...")
        #         message = await agent_client.messages.create(
        #             thread_id=thread_id,
        #             role="user",
        #             content=user_message.get('message', '')
        #         )
        #         logger.info(f"Created message, message ID: {message.id}")
        #     except Exception as e:
        #         logger.error(f"Error creating message: {e}")
        #         raise HTTPException(status_code=500, detail=f"Error creating message: {e}")

        # Set the Server-Sent Events (SSE) response headers.
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
        logger.info(f"Starting streaming response for thread ID {thread_id}")

        if message_from_cache:
            print(f"Usando mensaje de cache: {message_content}")
            #message_content = "Mensaje de prueba"
            response = StreamingResponse(string_streamer(message_content), headers=headers, media_type="text/event-stream")
        else:
            # Create the streaming response using the generator.
            response = StreamingResponse(get_result(request_str, request, thread_id, agent_id, ai_project, app_insights_conn_str, carrier), headers=headers)

        # Update cookies to persist the thread and agent IDs.
        response.set_cookie("thread_id", thread_id)
        response.set_cookie("agent_id", agent_id)
        
        print(f"El tipo de response es: {type(response)}")
        
        return response
    
def string_streamer(text: str):
    data = {
        "type": "completed_message", 
        "role": "assistant",
        "content": text
    }
    yield f"data: {json.dumps(data)}\n\n"

def read_file(path: str) -> str:
    with open(path, 'r') as file:
        return file.read()


def run_agent_evaluation(
    thread_id: str, 
    run_id: str,
    ai_project: AIProjectClient,
    app_insights_conn_str: str):

    if app_insights_conn_str:
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
        def run_evaluation():
            try:        
                logger.info(f"Running agent evaluation on thread ID {thread_id} and run ID {run_id}")
                agent_evaluation_response = ai_project.evaluations.create_agent_evaluation(
                    evaluation=agent_evaluation_request
                )
                logger.info(f"Evaluation response: {agent_evaluation_response}")
            except Exception as e:
                logger.error(f"Error creating agent evaluation: {e}")

        # Run the evaluation synchronously (no await)
        run_evaluation()


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