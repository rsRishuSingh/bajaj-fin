@tool
async def hybrid_search(self, question: str, top_k: int = 8) -> List[Dict]:
        """
        Performs a sophisticated hybrid search by combining lexical (BM25) and semantic (Vector)
        search results using Reciprocal Rank Fusion (RRF).

        This method encapsulates the core retrieval logic, acting as a powerful tool to find
        the most relevant context for a given query from the active document.

        Args:
            query: The user's natural language query.
            top_k: The final number of top results to return.

        Returns:
            A list of the most relevant chunk dictionaries.
        """
        if not self.active_bm25 or not self.active_chunks or not self.active_collection_name:
            print("Error: Cannot perform search. No document is active.")
            return []

        print(f"\n🔍 Performing Hybrid Search for: '{question}'")

        # 1. Expand Query using LLM for more comprehensive searching
        prompt_expand = f"""You are an expert document analyst. Your task is to deconstruct a user's conversational or vague question into a set of clear, specific, and standalone queries that can be used to search and extract relevant information from any structured or unstructured document. These queries should comprehensively cover all aspects of the user's original intent, such as definitions, scope, limitations, procedures, timelines, exclusions, and conditions—depending on the context of the question and nature of the document.

--- EXAMPLES ---

Original Question: I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?
Generated Queries:
Is cataract treatment a covered medical procedure?
What is the maximum limit or sub-limit payable for cataract surgery?
Are there specific conditions or waiting periods applicable to cataract treatment?
What are the policy's general exclusions related to eye surgeries or treatments?

Original Question: When will my root canal claim of Rs 25,000 be settled?
Generated Queries:
Is root canal treatment covered under the policy's dental benefits?
What is the process and typical timeline for claim settlement?
Are there monetary limits or sub-limits for outpatient dental procedures?
What are the waiting periods associated with dental treatments?

Original Question: Will this software support live video editing?
Generated Queries:
Does the software have live video editing capabilities?
Are there any performance or hardware requirements for live video editing?
Which file formats are supported for live editing?
Is there a limit to video resolution or length for real-time editing?

Original Question: Can I use this policy while traveling outside India?
Generated Queries:
Is international coverage included in this policy?
Are there any geographical exclusions or limitations?
What are the procedures for filing a claim from outside India?
Are there additional charges or riders for international usage?

--- YOUR TASK ---

Original Question: {question}
Generated Queries:
"""
        try:
            response = await self.llm.ainvoke(prompt_expand)
            expanded_queries = response.content.strip().split("\n")
            all_queries = [question] + [q.strip() for q in expanded_queries if q.strip()]
        except Exception:
            all_queries = [question]
        
        print(f"   - Expanded into {len(all_queries)} search queries.")

        # 2. Reciprocal Rank Fusion (RRF)
        # This dictionary will hold the fused scores for each document chunk.
        fused_scores = {}
        rrf_k = 60  # RRF constant, as used in the original script

        print("   - Fusing results from lexical and semantic search...")
        for q in all_queries:
            # Lexical Search (BM25)
            tokenized_query = q.lower().split()
            bm25_scores = self.active_bm25.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:15]
            for i, doc_id in enumerate(bm25_top_indices):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + i + 1)

            # Semantic Search (Qdrant)
            query_embedding = await self.embedder.aembed_query(q)
            semantic_results = await self.qdrant_client.search(
                collection_name=self.active_collection_name,
                query_vector=query_embedding,
                limit=15
            )
            for i, hit in enumerate(semantic_results):
                fused_scores[hit.id] = fused_scores.get(hit.id, 0) + 1 / (rrf_k + i + 1)

        # 3. Sort and Retrieve Top Chunks
        if not fused_scores:
            return []
            
        sorted_unique_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        retrieved_chunks = []
        for doc_id, score in sorted_unique_ids[:top_k]:
            if doc_id < len(self.active_chunks):
                retrieved_chunks.append(self.active_chunks[doc_id])

        print(f"   - Retrieved {len(retrieved_chunks)} relevant chunks after fusion.")
        return retrieved_chunks