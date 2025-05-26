import asyncio
from agno.agent import Agent
from agno.models.groq import Groq
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.pubmed import PubmedTools
from skin.skin_knowledge import DermaKnowledgeBase
from dotenv import load_dotenv
import os

load_dotenv()

agent_storage: str = "tmp/agents.db"

# ✅ Use this to initialize agents and serve the playground
async def main():
    # Load derma knowledge base
    kb = DermaKnowledgeBase(
        table_name="derma_knowledge",
        db_path="./my_local_lancedb",
        pdf_paths=["/home/wambui_nicole/VisionModel/resources/_OceanofPDF.com_Atlas_of_Clinical_Dermatology_in_Coloured_Skin_-_P_C_Das.pdf",
                   "/home/wambui_nicole/VisionModel/resources/Atlas of Dermatological Conditions in People of African Descent.pdf",
                   "/home/wambui_nicole/VisionModel/resources/Common Skin Diseases in Africa  an Illustrated Guide by Colette van Hees  Ben Naafs (z-lib.org).pdf"
        ],
        urls=[""]
    )
    await kb.aload(upsert=True, recreate=False)

    # Web agent
    web_agent = Agent(
        name="Web Agent",
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        tools=[DuckDuckGoTools()],
        instructions=["Always include sources"],
        storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=5,
        markdown=True,
    )

    # General medical agent
    med_agent = Agent(
        name="Medical Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[PubmedTools()],
        show_tool_calls=True,
        instructions=["Always include sources"],
        storage=SqliteStorage(table_name="med_agent", db_file=agent_storage),
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=5,
        markdown=True,
    )

    # Derma agent with loaded knowledge base
    derma_agent = Agent(
        name="Derma Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[PubmedTools()],
        knowledge_base=kb.get_knowledge_base(),
        show_tool_calls=True,
        instructions=["Always include sources"],
        storage=SqliteStorage(table_name="derma_agent", db_file="./derma_agent.sqlite"),
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=5,
        markdown=True,
    )

    # Launch the app with all agents
    app = Playground(agents=[web_agent, med_agent, derma_agent]).get_app()
    serve_playground_app(app, reload=True)

# Only run if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
