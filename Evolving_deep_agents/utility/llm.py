"""
Local Ollama LLM singleton used by the supervisor, codegen, and sub-agents.
"""
from langchain_ollama import ChatOllama

# Direct local Ollama channel (no '/v1' path suffix).
Model = ChatOllama(
    model="llama3.2:3b-instruct-q4_K_M",
    # model="gemma4:e4b",
    base_url="http://127.0.0.1:11434",
    temperature=0.0,
    num_ctx=65536,
    num_predict=32768,
    num_batch=128,
    # format="json",
)
