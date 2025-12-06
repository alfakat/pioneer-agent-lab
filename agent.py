from dataclasses import dataclass
from typing import List, Dict, Any

from sentence_transformers import util
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from text_search import chunks, corpus_embeddings, embedding_model


@dataclass
class Deps:
    chunks: List[Dict[str, Any]]
    corpus_embeddings: Any
    embedding_model: Any


deps = Deps(
    chunks=chunks,
    corpus_embeddings=corpus_embeddings,
    embedding_model=embedding_model)

# Make sure Ollama is running and you have: ollama pull llama3
ollama_model = OpenAIChatModel(
    model_name="llama3",
    provider=OllamaProvider(
        base_url="http://localhost:11434/v1"  # Ollama OpenAI-compatible endpoint),)

rag_agent = Agent(
    model=ollama_model,
    deps_type=Deps,
    system_prompt=(
        "You are a documentation assistant. "
        "You will receive some documentation context and a question. "
        "Answer ONLY using the given context. "
        "If the answer is not in the context, say you don't know."),)

def search_docs(
    query: str,
    deps: Deps,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search over repo chunks.
    Returns list of {score, text, filename, section_title}.
    """
    model = deps.embedding_model
    corpus_embeddings = deps.corpus_embeddings
    chunks = deps.chunks

    query_emb = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, corpus_embeddings)[0]

    top_k = min(top_k, len(chunks))
    top_results = cos_scores.topk(k=top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = int(idx)
        chunk = chunks[idx]
        meta = chunk["metadata"]

        results.append({
                "score": float(score),
                "text": chunk["text"],
                "filename": meta.get("filename"),
                "section_title": meta.get("section_title")})

    return results

if __name__ == "__main__":
    user_query = "information on prior major and minor releases"
    docs = search_docs(user_query, deps, top_k=5)
    context_parts: List[str] = []
    for d in docs:
        header = f"# {d['filename']} / {d['section_title']} (score={d['score']:.3f})"
        context_parts.append(header + "\n\n" + d["text"])

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "Use ONLY the documentation below to answer the question.\n\n"
        "=== DOCUMENTATION START ===\n"
        f"{context}\n"
        "=== DOCUMENTATION END ===\n\n"
        f"Question: {user_query}\n\n"
        "Answer:" )

    response = rag_agent.run_sync(prompt, deps=deps)

    print("\n=== FINAL ANSWER ===")
    print(response.output)
