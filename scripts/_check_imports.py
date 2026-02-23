"""Quick import smoke test for langchain 1.x compatibility."""
import sys

errors = []

tests = [
    ("langchain_classic.chains", ["create_history_aware_retriever", "create_retrieval_chain"]),
    ("langchain_classic.chains.combine_documents", ["create_stuff_documents_chain"]),
    ("langchain_classic.retrievers", ["EnsembleRetriever", "ContextualCompressionRetriever"]),
    ("langchain_classic.retrievers.document_compressors", ["CrossEncoderReranker"]),
    ("langchain_community.retrievers", ["BM25Retriever"]),
    ("langchain_community.cross_encoders", ["HuggingFaceCrossEncoder"]),
    ("langchain_google_genai", ["ChatGoogleGenerativeAI"]),
    ("langchain_huggingface", ["HuggingFaceEmbeddings"]),
    ("langchain_chroma", ["Chroma"]),
    ("langchain_core.prompts", ["ChatPromptTemplate", "MessagesPlaceholder"]),
    ("langchain_core.output_parsers", ["StrOutputParser"]),
]

for module, symbols in tests:
    try:
        mod = __import__(module, fromlist=symbols)
        for sym in symbols:
            getattr(mod, sym)
        print(f"  OK  {module}: {', '.join(symbols)}")
    except (ImportError, AttributeError) as e:
        print(f"  FAIL {module}: {e}")
        errors.append((module, str(e)))

print()
if errors:
    print(f"FAILED: {len(errors)} import(s) broken")
    sys.exit(1)
else:
    print("All imports OK")
