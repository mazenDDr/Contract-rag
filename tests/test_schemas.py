from contract_rag.schemas import RetrievalConfig


def test_config_id_is_stable_and_descriptive():
    cfg = RetrievalConfig(
        chunking="section", dense_model="bge-large", fusion="rrf", reranker="bge-reranker-v2-m3"
    )
    assert cfg.config_id() == "section__bm25__bge-large__rrf__bge-reranker-v2-m3__k8__doc"
    assert cfg.config_id() == RetrievalConfig(**cfg.model_dump()).config_id()
