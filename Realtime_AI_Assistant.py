from langchain_ollama import OllamaLLM  # Fix: New Ollama class
# from duckduckgo_search import DDGS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

llm = OllamaLLM(model="llama3.2")

# Built-in tool that handles searching and formatting automatically
search_tool = TavilySearchResults(max_results=5)

# def ddg_text_search(query: str) -> str:
#     """Simulates a browser search page for the LLM."""
#     try:
#         with DDGS() as ddgs:
#             # We search without a strict 'day' limit first to ensure we get results
#             results = ddgs.text(query, max_results=5)
#             if not results:
#                 return "No web results found."

#             # Join Title + Body so the LLM sees what you see in a browser
#             full_text = []
#             for r in results:
#                 line = f"Title: {r.get('title')}\nSnippet: {r.get('body')}"
#                 full_text.append(line)
#             return "\n\n".join(full_text)
#     except Exception:
#         return "Search failed."


# This tool will run a web search
# search = Tool(
#     name="duckduckgo_search",
#     description="A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events.",
#     func=ddg_text_search
# )

prompt = ChatPromptTemplate.from_template(
    """You are a real-time AI assistant. 
    Use the following search results to answer the user. 
    If the results contain a price or news, report it clearly.

    Search Results:
    {context}

    User Question: {question}
    """
)

# RAG chain using the search tool
chain = (
    RunnablePassthrough.assign(
        context=lambda x: search_tool.invoke({"query": x["question"]})
    )

    | prompt
    | llm
)

print("🤖 Hello! I'm a real-time AI assistant. What's new?")
while True:
    try:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("🤖 Goodbye!")
            break

        print("🤖 Thinking...")

        # Run the orchestration chain
        response = chain.invoke({"question": user_query})

        print(f"🤖: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")
