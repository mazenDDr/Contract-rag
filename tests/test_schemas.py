from contract_rag.schemas import RetrievalConfig


def test_config_id_is_stable_and_descriptive():
    cfg = RetrievalConfig(
        chunking="section", dense_model="bge-base", fusion="rrf", reranker="bge-reranker-base"
    )
    assert cfg.config_id() == "section__bm25__bge-base__rrf__bge-reranker-base__k8__doc"
    assert cfg.config_id() == RetrievalConfig(**cfg.model_dump()).config_id()
