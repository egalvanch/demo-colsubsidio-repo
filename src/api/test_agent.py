# test_print_agent.py
import os
from dotenv import load_dotenv
from azure.ai.agents import AgentsClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables from src/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

endpoint = os.environ.get("AZURE_AI_ENDPOINT")
key = os.environ.get("AZURE_AI_KEY")
agent_id = os.environ.get("AZURE_EXISTING_AGENT_ID", "asst_ShpnErTo1QD1qCOHlIUlTJXI")

if not endpoint or not key:
    raise RuntimeError("AZURE_AI_ENDPOINT and AZURE_AI_KEY must be set in src/.env")

client = AgentsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

agent = client.get_agent(agent_id)
print("Agent ID:", agent.id)
print("Model:", getattr(agent, "model", None))
print("Tools:", getattr(agent, "tools", None))
