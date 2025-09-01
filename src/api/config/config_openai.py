import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

AZURE_OPENAI_ENDPOINT = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_KEY = str(os.getenv("AZURE_OPENAI_KEY"))
AZURE_EMBEDDINGS_ENDPOINT = str(os.getenv("AZURE_EMBEDDINGS_ENDPOINT"))
