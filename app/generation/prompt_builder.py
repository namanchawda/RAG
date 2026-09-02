"""Prompt construction for grounded answering with SEC filing context."""


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Construct a grounded prompt using the retrieved filing chunks as context.

    The model is instructed to answer only with the supplied context and to refuse
    to guess when the answer is not present. Each chunk is labeled with its source
    file and chunk id so the model can cite which document section it used.
    """
    context_sections: list[str] = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source_file = chunk.get("source_file", "unknown")
        chunk_id = chunk.get("chunk_id", idx)
        text = chunk.get("chunk_text", "")
        context_sections.append(
            f"Context {idx} (source_file={source_file}, chunk_id={chunk_id})\n{text.strip()}"
        )

    context_block = "\n\n---\n\n".join(context_sections)

    return (
        "Answer the user's question ONLY using the provided context below. "
        "If the answer is not present in the provided context, say exactly: "
        '"I don\'t have enough information in the provided context to answer this." '
        "Do not use outside knowledge or guess.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )
