"""Find correct import paths in langchain 1.x."""
import importlib

candidates = {
    "create_history_aware_retriever": [
        ("langchain_classic.chains", "create_history_aware_retriever"),
        ("langchain.chains", "create_history_aware_retriever"),
    ],
    "create_retrieval_chain": [
        ("langchain_classic.chains", "create_retrieval_chain"),
        ("langchain.chains", "create_retrieval_chain"),
    ],
    "create_stuff_documents_chain": [
        ("langchain_classic.chains.combine_documents", "create_stuff_documents_chain"),
        ("langchain.chains.combine_documents", "create_stuff_documents_chain"),
    ],
    "EnsembleRetriever": [
        ("langchain_community.retrievers", "EnsembleRetriever"),
        ("langchain.retrievers", "EnsembleRetriever"),
    ],
    "ContextualCompressionRetriever": [
        ("langchain_community.retrievers", "ContextualCompressionRetriever"),
        ("langchain.retrievers", "ContextualCompressionRetriever"),
    ],
    "CrossEncoderReranker": [
        ("langchain_community.document_compressors", "CrossEncoderReranker"),
        ("langchain.retrievers.document_compressors", "CrossEncoderReranker"),
        ("langchain_classic.retrievers.document_compressors", "CrossEncoderReranker"),
    ],
}

for sym, paths in candidates.items():
    for mod_path, attr in paths:
        try:
            mod = importlib.import_module(mod_path)
            getattr(mod, attr)
            print(f"  FOUND  {sym}  →  from {mod_path} import {attr}")
            break
        except (ImportError, AttributeError):
            pass
    else:
        print(f"  MISSING {sym} — not found in any candidate path")
