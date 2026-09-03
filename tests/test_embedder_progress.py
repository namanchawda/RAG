from app.ingestion import embedder


def test_embed_texts_reports_after_each_small_batch(monkeypatch):
    calls = []
    progress = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(embedder, "load_embedding_model", lambda _: FakeModel())

    embeddings = embedder.embed_texts(
        ["chunk"] * 5,
        batch_size=2,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert len(embeddings) == 5
    assert [len(texts) for texts, _ in calls] == [2, 2, 1]
    assert [current for current, _ in progress] == [0, 2, 4, 5]
    assert all(kwargs["show_progress_bar"] is False for _, kwargs in calls)