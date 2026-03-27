from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import networkx as nx
import os

load_dotenv()

misral_api_key = os.getenv('Mistral_Api_Key')

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=misral_api_key
)

extract_prompt = PromptTemplate(
    template="""
You are an expert knowledge graph builder.
Extract entities and relationships from the text.
Return ONLY a JSON list. Each item must contain:
- "head": source entity
- "relation": relationship
- "tail": target entity

Text:
{text}

Output JSON:
""",
    input_variables=["text"]
)

extraction_chain = extract_prompt | llm | JsonOutputParser()

company_text = """
OpenAI was founded by Sam Altman and Elon Musk.
OpenAI developed GPT-4.
GPT-4 powers ChatGPT.
Microsoft partnered with OpenAI.
Microsoft invested 10 billion dollars in OpenAI.
ChatGPT is used by millions of users worldwide.
"""

triples = extraction_chain.invoke({'text': company_text})
# print(extraction_chain.invoke({'text': company_text}))

# 4. Build Knowledge Graph
kg = nx.DiGraph()  # DiGraph means "Directed Graph" (arrows point one way)


def build_knowledge_graph(triples):
    for item in triples:
        head = item.get("head")
        tail = item.get("tail")
        relation = item.get("relation")

        if head and tail:
            kg.add_node(head)
            kg.add_node(tail)
            kg.add_edge(head, tail, label=relation)


build_knowledge_graph(triples)

print("\n Nodes in Graph:")
# print(list(kg.nodes()))

# 5. MULTI-HOP RETRIEVAL


def retrieve_graph_context(entity, max_depth=2):
    context = set()
    visited_nodes = set()

    def dfs(node, depth):
        if depth > max_depth:
            return
        visited_nodes.add(node)

        # 1. Check Outgoing edges (What does this node do?)
        for neighbor in kg.successors(node):
            relation = kg.get_edge_data(node, neighbor)["label"]
            context.add(f"{node} {relation} {neighbor}")
            if neighbor not in visited_nodes:
                dfs(neighbor, depth + 1)

        # 2. Check Incoming edges (Who interacts with this node?)
        for predecessor in kg.predecessors(node):
            relation = kg.get_edge_data(predecessor, node)["label"]
            context.add(f"{predecessor} {relation} {node}")
            if predecessor not in visited_nodes:
                dfs(predecessor, depth + 1)

    if entity in kg.nodes:
        dfs(entity, 1)  # Start the traversal

    return ". ".join(context)


final_prompt = PromptTemplate(
    template="""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

rag_chain = final_prompt | llm

entity = 'ChatGPT'

graph_context = retrieve_graph_context(entity, max_depth=3)

print(f'Retreived Graph context: \n {graph_context}')

question = "Which company invested in the company that built ChatGPT?"

response = rag_chain.invoke({
    'context': graph_context,
    'question': question 
})

print("\n Final Answer:\n")
print(response.content)
