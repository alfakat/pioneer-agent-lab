from get_data import read_repo_data
from chunk_the_data import chunk_repo_docs
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
dtc_faq = read_repo_data('dbt-labs', 'dbt-core')
chunks = chunk_repo_docs(dtc_faq, max_chars=1200, overlap_chars=200)
corpus_texts = [c["text"] for c in chunks]

corpus_embeddings = embedding_model.encode(
    corpus_texts,
    convert_to_tensor=True)

query = "information on prior major and minor releases"
query_embedding = embedding_model.encode(
    query,
    convert_to_tensor=True)

cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

top_k = min(5, len(corpus_texts))
top_results = cos_scores.topk(k=top_k)

print(f"Query: {query}\n")
for score, idx in zip(top_results.values, top_results.indices):
    idx = int(idx)
    chunk = chunks[idx]
    print(f"Score: {float(score):.4f}")
    print(f"File: {chunk['metadata']['filename']}")
    print(f"Section: {chunk['metadata']['section_title']}")
    print("-" * 80)
    print(chunk["text"][:800])  # print first 800 chars
    print("=" * 80, "\n")

__all__ = ["chunks", "corpus_embeddings", "embedding_model"]
