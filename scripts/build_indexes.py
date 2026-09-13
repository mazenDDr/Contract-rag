"""Build BM25 and dense (Qdrant) indexes for every chunking strategy and embedding model."""

from contract_rag.retrieval.build import main

if __name__ == "__main__":
    main()
