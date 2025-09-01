import os
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, List
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorQuery
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
import ssl
import certifi

# Configurar certificados usando certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# ===========================
# Configuración desde .env
# ===========================
AI_PROJECT_ENDPOINT = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_AI_EMBED_DEPLOYMENT_NAME")
SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
INDEX_NAME = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_TARGET_EMBED_URI = os.getenv("AZURE_TARGET_EMBED_URI")

if not all([SEARCH_ENDPOINT, INDEX_NAME, SEARCH_API_KEY, AZURE_TARGET_EMBED_URI]):
    raise ValueError("Faltan variables de entorno requeridas para Azure AI Search o embeddings.")

# Cliente de Azure AI Search
if not SEARCH_ENDPOINT or not INDEX_NAME or not SEARCH_API_KEY:
    raise ValueError("Variables de entorno de Azure AI Search no configuradas correctamente")

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# ===========================
# Funciones principales
# ===========================

def get_embedding(text: str) -> List[float]:
    """
    Genera un embedding usando el deployment configurado en Azure OpenAI.
    """
    try:
        # Verificar que el URI esté configurado
        if not AZURE_TARGET_EMBED_URI:
            raise ValueError("AZURE_TARGET_EMBED_URI no está configurado")
        
        # Usar DefaultAzureCredential para obtener un token válido
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        # Usar el URI completo desde las variables de entorno
        url = AZURE_TARGET_EMBED_URI
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token.token}"
        }
        payload = {"input": text}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        raise Exception(f"Error generando embedding: {e}")

def hybrid_search(question: str, embedding: List[float], k: int = 3, threshold: float = 0.0331) -> Optional[str]:
    """
    Realiza una búsqueda híbrida (texto + vector) en Azure AI Search.
    Devuelve la respuesta más relevante si supera el umbral.
    """
    vector_query = VectorQuery(
        k_nearest_neighbors=k,
        fields="embedding",
        vector=embedding,
        kind="vector"
    )
    results = search_client.search(search_text=question, vector_queries=[vector_query], top=k)
    best_doc, best_score = None, float("-inf")
    for doc in results:
        score = doc.get("@search.score", 0.0)
        if score > best_score:
            best_doc, best_score = doc, score
    return best_doc.get("answer") if best_doc and best_score >= threshold else None

def upsert_qa(question: str, answer: str):
    """
    Inserta o actualiza un documento en el índice de Azure AI Search.
    """
    embedding = get_embedding(question)
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "id": str(abs(hash(question))),
        "question": question,
        "embedding": embedding,
        "answer": answer,
        "lastUpdated": now
    }
    try:
        search_client.merge_or_upload_documents(documents=[doc])
    except HttpResponseError as e:
        print("Error al indexar:", e)
        raise
