import os
import asyncio
from azure.identity.aio import DefaultAzureCredential
from azure.ai.agents.aio import AgentsClient
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

async def main():
    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    agent_id = os.getenv("AZURE_EXISTING_AGENT_ID")
    credential = DefaultAzureCredential()

    client = AgentsClient(endpoint=endpoint, credential=credential)
    thread = await client.threads.create()
    await client.messages.create(thread_id=thread.id, role="user", content="¿Cuáles son los beneficios de salud de Colsubsidio?")
    run = await client.runs.create(thread_id=thread.id, agent_id=agent_id)
    result = await client.runs.get(run_id=run.id, thread_id=thread.id)
    print(result)

asyncio.run(main())