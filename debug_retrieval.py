# debug_retrieval.py (create in root folder)
from dotenv import load_dotenv
load_dotenv()

from src.retriever import load_vectorstore

vs = load_vectorstore()

# Direct similarity search — bypass classification entirely
query = "What is LCEL LangChain Expression Language?"
results = vs.similarity_search(query, k=4)

print(f"\nQuery: {query}")
print("="*60)
for i, doc in enumerate(results):
    print(f"\n[{i+1}] source: {doc.metadata.get('source', '')}")
    print(f"     doc_type: {doc.metadata.get('doc_type', '')}")
    print(f"     preview: {doc.page_content[:200]}")