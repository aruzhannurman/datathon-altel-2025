#!/usr/bin/env python3
"""
Simple Clean RAG System - Q/A/C Format
Systematic approach: Query Enhancement → Semantic + BM25 → LLM Reranking → Answer Generation
"""

import json
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import os
import warnings

warnings.filterwarnings("ignore")

# Try to import dependencies
try:
    from sentence_transformers import SentenceTransformer

    SEMANTIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SentenceTransformer not available: {e}")
    SEMANTIC_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        openai_client = OpenAI()
except ImportError:
    OPENAI_AVAILABLE = False


class SimpleCleanRAG:
    def __init__(self):
        self.chunks = []
        self.bm25 = None
        self.semantic_available = False
        self.semantic_index = None
        self.semantic_metadata = []
        self.sentence_model = None

        # Load data and initialize
        self.load_chunks()
        self.initialize_bm25()
        self.initialize_semantic()

    def load_chunks(self):
        """Load chunks from the unified database metadata."""
        try:
            metadata_path = "models/markdown_vector_databases/unified_markdown_database_metadata.json"
            with open(metadata_path, "r", encoding="utf-8") as f:
                database_info = json.load(f)
                self.chunks = database_info["chunks_metadata"]
            # print(f"✅ Loaded {len(self.chunks)} chunks")
        except Exception as e:
            print(f"❌ Failed to load chunks: {e}")

    def initialize_bm25(self):
        """Initialize BM25 search."""

        def tokenize(text):
            # Simple tokenization
            return re.findall(r"\b\w+\b", text.lower())

        tokenized_docs = [tokenize(chunk.get("text", "")) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.tokenize = tokenize
        # print("✅ BM25 initialized")

    def initialize_semantic(self):
        """Initialize semantic search using the existing FAISS database."""
        if not SEMANTIC_AVAILABLE:
            self.semantic_available = False
            return

        try:
            db_path = "data/markdown_vector_databases"
            faiss_path = os.path.join(db_path, "unified_markdown_database.faiss")
            metadata_path = os.path.join(
                db_path, "unified_markdown_database_metadata.json"
            )

            if os.path.exists(faiss_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.semantic_index = faiss.read_index(faiss_path)

                # Load metadata
                with open(metadata_path, "r", encoding="utf-8") as f:
                    database_info = json.load(f)
                    self.semantic_metadata = database_info["chunks_metadata"]

                # Check if we should use OpenAI or SentenceTransformer based on FAISS dimensions
                if self.semantic_index.d == 3072:
                    # FAISS was created with OpenAI embeddings
                    self.embedding_method = "openai"
                    if OPENAI_AVAILABLE:
                        pass
                    else:
                        # print("❌ FAISS database requires OpenAI embeddings, but OpenAI API key not available")
                        self.semantic_available = False
                        return
                elif self.semantic_index.d == 384:
                    # FAISS was created with SentenceTransformer
                    self.embedding_method = "sentence_transformer"
                    self.sentence_model = SentenceTransformer(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    # print("🔧 Using SentenceTransformer embeddings to match existing FAISS database")
                else:
                    # print(f"❌ Unknown embedding dimension: {self.semantic_index.d}")
                    self.semantic_available = False
                    return

                self.semantic_available = True
                # print(f"✅ Semantic search initialized with {len(self.semantic_metadata)} embeddings")
            else:
                # print("⚠️ FAISS database not found")
                self.semantic_available = False

        except Exception as e:
            print(f"❌ Failed to load semantic search: {e}")
            self.semantic_available = False

    def extract_entities(self, query: str) -> List[str]:
        """Extract key entities (countries, services) from query using LLM."""
        if not OPENAI_AVAILABLE:
            return []

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Extract key entities from this telecom customer service query.

Query: "{query}"

Please identify and extract:
1. Country names (in all forms: nominative, genitive, etc.) - e.g., "Турции" should extract both "Турции" and "Турция"
2. Service types - roaming, tariff, internet, SMS, calls, etc.
3. Company names - Altel, Tele2, etc.

For each country mentioned, provide both the original form and the standard nominative form.

Respond ONLY with valid JSON, no additional text:
{{
    "entities": ["entity1", "entity2", "entity3"]
}}""",
                    }
                ],
                temperature=0.1,
            )

            # Clean the response content to ensure it's valid JSON
            content = response.choices[0].message.content.strip()

            # Remove any markdown formatting if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()

            result = json.loads(content)
            return result.get("entities", [])

        except Exception as e:
            print(f"Entity extraction failed: {e}")
            return []

    def enhance_query(self, query: str) -> tuple:
        """Enhance query using LLM to improve search performance."""
        if not OPENAI_AVAILABLE:
            return query, [query]

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are helping improve search queries for a Kazakhstani telecom company (Altel/Tele2) customer service system.

Original query: "{query}"

Please provide:
1. An enhanced/corrected version of the query (fix typos, expand abbreviations, add context)
2. A list of 3-5 key search terms/phrases that would help find relevant information

Focus on telecom services like: verification, activation, roaming, internet settings, office addresses, salon codes, device registration, tariffs, etc.

Respond ONLY with valid JSON, no additional text:
{{
    "enhanced_query": "improved version of the query",
    "search_terms": ["term1", "term2", "term3"]
}}""",
                    }
                ],
                temperature=0.1,
            )

            # Clean the response content to ensure it's valid JSON
            content = response.choices[0].message.content.strip()

            # Remove any markdown formatting if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()

            result = json.loads(content)
            return result.get("enhanced_query", query), result.get(
                "search_terms", [query]
            )

        except Exception as e:
            print(f"Query enhancement failed: {e}")
            return query, [query]

    def bm25_search(
        self, query: str, top_k: int = 20, extracted_entities: List[str] = None
    ) -> List[Dict[str, Any]]:
        """BM25 search with boost for exact entity matches."""
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        results = []
        query_lower = query.lower()

        # Use extracted entities if provided, otherwise extract them
        if extracted_entities is None:
            extracted_entities = self.extract_entities(query)

        for i, score in enumerate(scores):
            if score > 0 and i < len(self.chunks):
                chunk = self.chunks[i]
                chunk_text_lower = chunk.get("text", "").lower()

                # Boost score for entity matches
                boost_factor = 1.0

                # Check for entity matches
                for entity in extracted_entities:
                    entity_lower = entity.lower()
                    if entity_lower in chunk_text_lower:
                        boost_factor = 3.0  # Significant boost for entity matches
                        break

                # Additional boost for structured pricing data
                if any(
                    indicator in chunk_text_lower
                    for indicator in ["₸", "тенге", "за минуту", "за мб", "за штуку"]
                ):
                    boost_factor *= 1.5

                boosted_score = score * boost_factor

                results.append(
                    {
                        "text": chunk.get("text", ""),
                        "source_file": chunk.get("source_file", ""),
                        "language": chunk.get("language", "unknown"),
                        "content_type": chunk.get("content_type", "text"),
                        "bm25_score": float(boosted_score),
                        "search_method": "bm25",
                    }
                )

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Semantic search using the existing FAISS database."""
        if not self.semantic_available:
            return []

        try:
            # Generate query embedding based on the method used to create FAISS
            if self.embedding_method == "openai":
                # Use OpenAI embeddings
                response = openai_client.embeddings.create(
                    input=[query], model="text-embedding-3-large"
                )
                query_embedding = np.array(
                    [response.data[0].embedding], dtype="float32"
                )
            else:
                # Use SentenceTransformer embeddings
                query_embedding = self.sentence_model.encode([query]).astype("float32")

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search in FAISS index
            scores, indices = self.semantic_index.search(query_embedding, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.semantic_metadata):
                    metadata = self.semantic_metadata[idx]
                    results.append(
                        {
                            "text": metadata.get("text", ""),
                            "source_file": metadata.get("source_file", ""),
                            "language": metadata.get("language", "unknown"),
                            "content_type": metadata.get("content_type", "text"),
                            "semantic_score": float(score),
                            "search_method": "semantic",
                        }
                    )

            return results

        except Exception as e:
            print(f"Semantic search failed: {e}")
            return []

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using word overlap."""
        words1 = set(self.tokenize(text1))
        words2 = set(self.tokenize(text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def search(
        self,
        original_query: str,
        enhanced_query: str,
        search_terms: List[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Combined search using both original and enhanced queries."""
        all_results = []

        # Extract key entities using LLM
        key_entities = self.extract_entities(original_query)

        # Search with original query
        all_results.extend(self.bm25_search(original_query, top_k, key_entities))
        if self.semantic_available:
            all_results.extend(self.semantic_search(original_query, top_k))

        # Search with enhanced query
        if enhanced_query != original_query:
            all_results.extend(self.bm25_search(enhanced_query, top_k, key_entities))
            if self.semantic_available:
                all_results.extend(self.semantic_search(enhanced_query, top_k))

        # Search with key entities (important for exact matches)
        for entity in key_entities:
            all_results.extend(self.bm25_search(entity, top_k, key_entities))
            if self.semantic_available:
                all_results.extend(self.semantic_search(entity, top_k))

        # Search with individual terms
        for term in search_terms[:3]:  # Limit to avoid too many results
            if term not in [original_query, enhanced_query]:
                all_results.extend(self.bm25_search(term, 5, key_entities))
                if self.semantic_available:
                    all_results.extend(self.semantic_search(term, 5))

        # Remove duplicates and combine scores
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            # Check for exact duplicates
            if text in seen_texts:
                # Combine scores from different methods
                existing = seen_texts[text]
                existing["combined_score"] = (
                    existing.get("combined_score", 0)
                    + result.get("bm25_score", 0)
                    + result.get("semantic_score", 0)
                )
            else:
                # Check for very similar content (>85% similarity)
                is_duplicate = False
                for existing_text in seen_texts.keys():
                    similarity = self._calculate_text_similarity(text, existing_text)
                    if similarity > 0.85:  # 85% similarity threshold
                        # Combine with existing similar text
                        existing = seen_texts[existing_text]
                        existing["combined_score"] = (
                            existing.get("combined_score", 0)
                            + result.get("bm25_score", 0)
                            + result.get("semantic_score", 0)
                        )
                        is_duplicate = True
                        break

                if not is_duplicate:
                    result["combined_score"] = result.get("bm25_score", 0) + result.get(
                        "semantic_score", 0
                    )
                    seen_texts[text] = result

        # Sort by combined score and return top results
        final_results = list(seen_texts.values())
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return final_results[:top_k]

    def rerank_with_llm(
        self, query: str, results: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank results using LLM."""
        if not OPENAI_AVAILABLE or not results:
            return results[:top_k]

        try:
            # Prepare documents for reranking
            docs_text = ""
            for i, result in enumerate(results[:10]):  # Limit to top 10 for LLM
                docs_text += f"Document {i+1}:\n{result['text'][:500]}...\n\n"

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are helping rank search results for a telecom customer service query.

Query: "{query}"

Documents:
{docs_text}

Please rank these documents by relevance to the query. Use this priority order:

**HIGHEST PRIORITY (rank first):**
1. Documents that contain EXACT MATCHES for specific entities mentioned in the query (country names, service names, operator names, etc.)
2. Documents with specific pricing/tariff information for the exact entity requested

**HIGH PRIORITY:**
3. Documents with complete actionable information that directly answers the question
4. Documents with structured data like pricing tables, specific numbers, step-by-step instructions

**LOWER PRIORITY:**
5. Documents with general information about the topic
6. Documents that mainly contain URLs or require external searches

**CRITICAL:** If the query mentions a specific country, service, or entity, prioritize documents that explicitly mention that same entity, even if other documents have more detailed information about different entities.

Example: For "роуминг в Турции" - prioritize documents mentioning "Турция" over documents with detailed pricing for other countries.

Respond ONLY with valid JSON, no additional text:
{{"rankings": [1, 3, 2, ...]}}""",
                    }
                ],
                temperature=0.1,
            )

            # Clean the response content to ensure it's valid JSON
            content = response.choices[0].message.content.strip()

            # Remove any markdown formatting if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()

            rankings_data = json.loads(content)
            rankings = rankings_data.get(
                "rankings", list(range(1, len(results[:10]) + 1))
            )

            # Reorder results based on LLM rankings
            reranked = []
            for rank in rankings[:top_k]:
                if 1 <= rank <= len(results):
                    result = results[rank - 1].copy()
                    result["llm_rank"] = len(rankings) - rankings.index(rank) + 1
                    reranked.append(result)

            return reranked

        except Exception as e:
            print(f"LLM reranking failed: {e}")
            return results[:top_k]

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate natural language answer using LLM."""
        if not OPENAI_AVAILABLE or not chunks:
            return "I don't have enough information to answer this question."

        try:
            # Prepare context from chunks
            context = ""
            for i, chunk in enumerate(chunks[:3], 1):
                context += f"Source {i}:\n{chunk['text']}\n\n"

            # Determine language
            is_kazakh = any(char in query for char in "әіңғүұқөһ")
            language = "Kazakh" if is_kazakh else "Russian"

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a helpful customer service assistant for Altel and Tele2 telecom companies in Kazakhstan.

User question: "{query}"

Context information:
{context}

Please provide a helpful answer based on the context information provided above. Follow these guidelines:
- Respond in {language}
- Be polite and professional
- Be concise but complete
- Use at most one emoji at the end
- Do NOT use any text formatting like bold, italic, or markdown
- Write in plain text only
- Answer based on the context provided - if relevant information is in the context, use it to answer

Answer:""",
                    }
                ],
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Answer generation failed: {e}")
            return "I apologize, but I'm unable to process your question right now."

    def query(self, question: str) -> Dict[str, Any]:
        """Main query function - returns Q/A/C format."""
        # print(f"\n🔍 Processing query: {question}")

        # Step 1: Enhance query
        enhanced_query, search_terms = self.enhance_query(question)
        if enhanced_query != question:
            pass

        # Step 2: Search (BM25 + Semantic)
        results = self.search(question, enhanced_query, search_terms, top_k=15)
        # print(f"🔎 Found {len(results)} initial results")

        # Step 3: LLM Reranking
        reranked_results = self.rerank_with_llm(question, results, top_k=3)
        # print(f"🤖 Reranked to top {len(reranked_results)} results")

        # Step 4: Generate answer
        answer = self.generate_answer(question, reranked_results)

        return {"question": question, "answer": answer, "chunks": reranked_results}


def main():
    # print("🚀 Initializing Simple Clean RAG System...")
    rag = SimpleCleanRAG()

    # print("\n" + "="*50)
    # print("Simple Clean RAG System Ready!")
    # print("Available methods: Semantic Search + BM25 + LLM Reranking")
    # print("="*50)

    while True:
        try:
            question = input("\nEnter your question (or 'quit' to exit): ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                # print("👋 Goodbye!")
                break

            if not question:
                continue

            # Get Q/A/C result
            result = rag.query(question)
            final_answer = result["answer"]
            final_answer = final_answer.replace("\n", " ").replace("\r", "")

            # Display in Q/A/C format
            print(f"\nQ: {result['question']}")
            print(f"A: {final_answer}")
            # print("C:")

            # for i, chunk in enumerate(result['chunks'], 1):
            #     print(f"  {i}. {chunk['text']}")
            #     print(f"     Source: {chunk['source_file']} | Method: {chunk.get('search_method', 'combined')}")
            #     if i < len(result['chunks']):
            #         print()

        except KeyboardInterrupt:
            # print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
