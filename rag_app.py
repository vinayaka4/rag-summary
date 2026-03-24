import sys

from dotenv import load_dotenv

load_dotenv()

from rag_core import (
    answer_question,
    build_vectorstore,
    default_embeddings,
    default_llm,
    load_documents,
    require_openai_key,
)

try:
    require_openai_key()
except RuntimeError as e:
    print(e)
    sys.exit(1)

# Example: point at a local PDF (or change path to a .docx / .xlsx)
PDF_PATH = r"C:\Users\91903\Desktop\Projects\Rag\Pdp-skimmer\Embodied Asymmetric Two-Handed Interactions for Immersive Data Exploration.pdf"

documents = load_documents(PDF_PATH)
embeddings = default_embeddings()
db = build_vectorstore(documents, embeddings)
llm = default_llm()

while True:
    query = input("Ask question: ")
    print("\nAnswer:\n", answer_question(db, llm, query))
