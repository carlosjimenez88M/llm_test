### Describe your approach to optimizing prompts for a Retrieval-Augmented Generation system. How would you maximize the retrieval of relevant information?


To optimize prompts for a Retrieval-Augmented Generation (RAG) system, my approach focuses on leveraging advanced retrieval and reranking techniques to maximize the relevance of retrieved information:

1. **Efficient Retrieval**: Use vector search methods with embeddings stored in vector databases, ensuring semantic similarity between queries and documents. Combine this with hybrid retrieval (e.g., BM25 + embeddings) for robust initial document selection.

2. **Reranking for Precision**: Implement learning-to-rank (LTR) models or cross-encoders. LTR models prioritize documents based on relevance scores learned from labeled data, while cross-encoders enhance ranking by evaluating semantic proximity between query-document pairs.

3. **Iterative Refinement**: Apply Maximum Marginal Relevance (MMR) to balance relevance and diversity in retrieved results. This ensures a varied yet contextually aligned set of documents is passed to the LLM.

4. **LLM-Informed Ranking**: Use the LLM itself as a reranker, either by generative ranking or feedback-based extractive methods, allowing the LLM to refine results based on task-specific intent and context.

This combination ensures highly relevant, diverse, and contextually appropriate information feeds into the LLM, enhancing response quality and system effectiveness.


```{python}
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Step 1: Prepare a Vector Database (FAISS) and Document Embeddings
# Using excerpts from "Twenty Thousand Leagues Under the Sea" by Jules Verne
documents = [
    "The sea is everything. It covers seven tenths of the terrestrial globe.",
    "The Nautilus was now floating in the midst of a phosphorescent bed.",
    "We may brave human laws, but we cannot resist natural ones.",
    "I am Captain Nemo, and you are my guest aboard the Nautilus.",
    "Science, my boy, is made up of mistakes, but they are mistakes which it is useful to make.",
    "The sea does not belong to despots. Upon its surface, men can still exercise unjust laws, fight, tear one another to pieces, and be carried away with terrestrial horrors."
]

query = "What is the sea?"

# Load pre-trained SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

# Encode documents and query
document_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

# Create FAISS Index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings.cpu().numpy())

# Step 2: Retrieve Top-k Documents
k = 3  
distances, indices = index.search(query_embedding.cpu().numpy().reshape(1, -1), k)
retrieved_docs = [documents[i] for i in indices[0]]

print("Retrieved Documents:")
for doc in retrieved_docs:
    print(f"- {doc}")

# Step 3: Rerank Using Maximum Marginal Relevance (MMR)
def mmr(query_embedding, doc_embeddings, docs, lambda_param=0.7, top_n=2):
    selected_docs = []
    remaining_indices = list(range(len(docs)))

    while len(selected_docs) < top_n and remaining_indices:
        mmr_scores = []
        for idx in remaining_indices:
            doc_similarity = util.cos_sim(query_embedding, doc_embeddings[idx]).item()
            redundancy = max(
                [util.cos_sim(doc_embeddings[idx], doc_embeddings[sel_idx]).item() for sel_idx in selected_docs] \
                if selected_docs else [0]
            )
            mmr_score = lambda_param * doc_similarity - (1 - lambda_param) * redundancy
            mmr_scores.append((mmr_score, idx))

        best_doc = max(mmr_scores, key=lambda x: x[0])[1]
        selected_docs.append(best_doc)
        remaining_indices.remove(best_doc)

    return [docs[idx] for idx in selected_docs]

reranked_docs = mmr(query_embedding, document_embeddings, documents)

print("\nReranked Documents:")
for doc in reranked_docs:
    print(f"- {doc}")

# Step 4: Pass to LLM for Final Refinement
llm = pipeline("text2text-generation", model="google/flan-t5-base")

prompt = f"Given the following context: {reranked_docs}, answer the query: '{query}'"
response = llm(prompt, max_length=100)

print("\nLLM Response:")
print(response[0]['generated_text'])
#LLM Response:
#The sea is described as everything, covering most of the Earth's surface and symbolizing freedom from terrestrial conflicts and injustices.

```

