# import json
# import os
# import time

# from dotenv import load_dotenv
# from openai import OpenAI
# from pageindex import PageIndexClient

# load_dotenv()

# PAGEINDEX_API_KEY = "6c2739b0ebba162168b47f0e2e"
# OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

# print("PageIndex key loaded:", "✅" if PAGEINDEX_API_KEY else "❌ Missing!")
# print("OpenAI key loaded:   ", "✅" if OPENAI_API_KEY    else "❌ Missing!")


# pi_client     = PageIndexClient(api_key=PAGEINDEX_API_KEY)
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# print("✅ PageIndex client ready")
# print("✅ OpenAI client ready")


# # ── Upload your PDF ─────────────────────────────────────────────────────────
# # Great candidates: Annual reports, research papers, legal docs, textbooks

# PDF_PATH = "../data"   # ← change this

# print(f"📤 Uploading: {PDF_PATH}")
# result = pi_client.submit_document(PDF_PATH)
# doc_id = result["doc_id"]

# print("✅ Uploaded!")
# print(f"📋 Document ID: {doc_id}")
# print("   (Save this ID — you'll use it throughout the notebook)")


# # ── Poll until processing is complete ───────────────────────────────────────
# # PageIndex builds the tree asynchronously.
# # For a 50-page PDF this typically takes 30–90 seconds.

# print("⏳ Building tree index...")
# print("   (This runs once per document — the index is cached for reuse)")

# while True:
#     status_result = pi_client.get_document(doc_id)
#     status = status_result.get("status")
#     print(f"   Status: {status}")
    
#     if status == "completed":
#         print("\n✅ Tree index ready!")
#         break
#     elif status == "failed":
#         print("\n❌ Processing failed. Check your PDF format.")
#         break
    
#     time.sleep(5)


# # ── Fetch the full tree ─────────────────────────────────────────────────────
# tree_result  = pi_client.get_tree(doc_id, node_summary=True)
# pageindex_tree = tree_result.get("result", [])

# print(f"📊 Top-level sections: {len(pageindex_tree)}")
# print("\n🌲 Raw tree (first node):")
# print(json.dumps(pageindex_tree[0] if pageindex_tree else {}, indent=2))


# # ── Pretty-print the full tree ───────────────────────────────────────────────
# def print_tree(nodes, indent=0):
#     """Recursively print tree titles for a visual overview."""
#     for node in nodes:
#         prefix = "  " * indent + ("└─ " if indent > 0 else "")
#         page   = node.get("page_index", "?")
#         print(f"{prefix}[{node['node_id']}] {node['title']}  (p.{page})")
#         if node.get("nodes"):
#             print_tree(node["nodes"], indent + 1)

# print("📚 Full Document Structure:\n")
# print_tree(pageindex_tree)


# # ── Count total nodes ────────────────────────────────────────────────────────
# def count_nodes(nodes):
#     total = len(nodes)
#     for n in nodes:
#         if n.get("nodes"):
#             total += count_nodes(n["nodes"])
#     return total

# total = count_nodes(pageindex_tree)
# print(f"🔢 Total nodes in tree: {total}")
# print("   Each node = one retrievable section of the document")


# # ── LLM Tree Search Function ─────────────────────────────────────────────────

# def llm_tree_search(query: str, tree: list, model: str = "gpt-4o") -> dict:
#     """
#     Core PageIndex retrieval:
#     Sends the query + document tree to an LLM.
#     LLM reasons over the structure and returns relevant node_ids.
    
#     Returns: dict with 'thinking' (reasoning) and 'node_list' (node IDs)
#     """
    
#     # Compress tree to save tokens — only send titles + short summaries
#     def compress(nodes):
#         out = []
#         for n in nodes:
#             entry = {
#                 "node_id": n["node_id"],
#                 "title":   n["title"],
#                 "page":    n.get("page_index", "?"),
#                 "summary": n.get("text", "")[:150]  # first 150 chars
#             }
#             if n.get("nodes"):
#                 entry["children"] = compress(n["nodes"])
#             out.append(entry)
#         return out
    
#     compressed_tree = compress(tree)
    
#     prompt = f"""You are given a query and a document's tree structure (like a Table of Contents).
#     Your task: identify which node IDs most likely contain the answer to the query.
#     Think step-by-step about which sections are relevant.

#     Query: {query}

#     Document Tree:
#     {json.dumps(compressed_tree, indent=2)}

#     Reply ONLY in this exact JSON format:
#     {{
#     "thinking": "<your step-by-step reasoning>",
#     "node_list": ["node_id1", "node_id2"]
#     }}"""

#     response = openai_client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"}
#     )
    
#     return json.loads(response.choices[0].message.content)


# # ── Test with a sample query ─────────────────────────────────────────────────
# query = "What is the syllabus covered in Modern LLM finetuning?"

# print(f"🔍 Query: {query}\n")
# result = llm_tree_search(query, pageindex_tree)

# print("🧠 LLM Reasoning:")
# print(result.get("thinking", "N/A"))
# print()
# print("🎯 Selected Node IDs:", result.get("node_list", []))


# # ── Generate answer from retrieved nodes ─────────────────────────────────────

# def generate_answer(query: str, nodes: list, model: str = "gpt-4o") -> str:
#     """
#     Takes retrieved nodes as context and generates a grounded answer.
#     Instructs the LLM to cite section titles and page numbers.
#     """
#     if not nodes:
#         return "⚠️ No relevant sections found in the document."
    
#     # Build context string from retrieved nodes
#     context_parts = []
#     for node in nodes:
#         context_parts.append(
#             f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
#             f"{node.get('text', 'Content not available.')}"
#         )
#     context = "\n\n---\n\n".join(context_parts)
    
#     prompt = f"""You are an expert document analyst.
#     Answer the question using ONLY the provided context.
#     For every claim you make, cite the section title and page number in parentheses.
#     Be concise and precise.

#     Question: {query}

#     Context:
#     {context}

#     Answer:"""
    
#     response = openai_client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}]
#     )
    
#     return response.choices[0].message.content


# # ── The complete Vectorless RAG function ─────────────────────────────────────

# def vectorless_rag(query: str, tree: list, verbose: bool = True) -> str:
#     """
#     Full end-to-end PageIndex RAG pipeline:
    
#     Step 1: LLM Tree Search  → finds relevant node_ids
#     Step 2: Node Retrieval   → fetches section content
#     Step 3: Answer Generation → produces cited answer
#     """
#     if verbose:
#         print(f"{'='*55}")
#         print(f"🔍 Query: {query}")
#         print(f"{'='*55}")
    
#     # Step 1: Tree Search
#     search_result  = llm_tree_search(query, tree)
#     node_ids       = search_result.get("node_list", [])
    
#     if verbose:
#         print(f"\n🧠 Reasoning: {search_result.get('thinking', '')[:200]}...")
#         print(f"🎯 Retrieved node IDs: {node_ids}")
    
#     # Step 2: Retrieve nodes
#     nodes = find_nodes_by_ids(tree, node_ids)
    
#     if verbose:
#         print(f"📄 Sections found: {[n['title'] for n in nodes]}")
    
#     # Step 3: Generate answer
#     answer = generate_answer(query, nodes)
    
#     if verbose:
#         print(f"\n📝 Answer:\n{answer}")
    
#     return answer


# # ── Run the full pipeline ────────────────────────────────────────────────────
# answer = vectorless_rag(
#     query="What are the syllabus covered in modern llm finetuning?",
#     tree=pageindex_tree
# )


# # ── Test with multiple queries ───────────────────────────────────────────────
# test_queries = [
#     "What are the syllabus covered in modern llm finetuning?",
#     "What are the syllabus covered in RAG?",
#     "Summarize the syllabus of Tokenization Deep Dive?",
# ]

# for q in test_queries:
#     print()
#     ans = vectorless_rag(q, pageindex_tree, verbose=False)
#     print(f"Q: {q}")
#     print(f"A: {ans[:300]}...")
#     print("-" * 55)







# # ── Define domain expert rules ───────────────────────────────────────────────
# # These are routing rules that tell the LLM WHERE to look for specific queries.
# # Think of it as encoding a senior analyst's institutional knowledge.


# FINANCIAL_EXPERT_RULES = """
# Expert routing rules for financial documents (10-K, annual reports):
# - EBITDA, profitability queries    → MD&A section (Management Discussion & Analysis)
# - Liquidity, cash flow queries     → Cash Flow Statement + liquidity footnotes
# - Risk factor queries              → Part I, Item 1A (Risk Factors)  
# - Revenue breakdown queries        → Segment reporting or Item 7
# - Forward-looking / strategy       → CEO letter, Outlook, Strategy section
# - Debt, credit, leverage queries   → Balance Sheet + debt footnotes
# - Regulatory / compliance queries  → Legal Proceedings or regulatory filings
# """

# print("✅ Expert rules defined")
# print("   These get injected into the retrieval prompt at query time.")


# # ── Expert Routing Rules — Advanced Route of Learning AI ─────────────────────
# # Krish Naik Academy | 21 Modules | 38 Sections | 481 Topics
# FINANCIAL_EXPERT_RULES = """
# Route queries to the correct module using these rules:
 
# M1  Neural Network Refresher   → backprop, activations, optimizers, PyTorch basics
# M2  Hardware                   → GPU, TPU, Apple Silicon, compute infrastructure
# M3  Transformers 101           → attention, self-attention, encoder-decoder, MHA
# M4  Tokenization               → BPE, WordPiece, SentencePiece, Byte Latent Transformers
# M5  Finetuning Architectures   → hands-on BERT/GPT/T5 finetuning, Hugging Face
# M6  KV Cache & Attention       → KV cache, Flash Attention, MQA, GQA, RoPE, vLLM
# M7  Scaling Laws               → Kaplan, Chinchilla, compute-optimal training
# M8  Mixture of Experts         → MoE, sparse computation, Mixture of Depths
# M9  Modern LLM Finetuning      → LoRA, QLoRA, SFT, DPO, PPO, RLHF, GRPO, ORPO,
#                                   quantization, TRL, Unsloth, synthetic data,
#                                   reasoning models, evaluation, deployment
# M10 SLM                        → small language models, pruning, when SLM vs LLM
# M11 Knowledge Distillation     → student-teacher, soft labels, DistilBERT, DeepSeek-R1
# M12 Hybrid Architectures       → Mamba, RWKV, SSMs, Jamba, Nemotron, beyond Transformers
# M13 Vision Foundations         → ViT, patch embeddings, CLIP, SigLIP, DINOv2
# M14 Visual Language Models     → VLM architecture, aligner, multimodal reasoning
# M15 Stable Diffusion & DiT     → DDPM, latent diffusion, FLUX.1, ControlNet, DreamBooth
# M16 Embedding Models           → dense, sparse, binary, Matryoshka, MRL, fine-tuning
# M17 RAG                        → chunking, BM25, ColBERT, hybrid RAG, rerankers,
#                                   self/corrective/adaptive/agentic RAG, Graph RAG,
#                                   multi-modal RAG, ColPali, RAG security
# M18 Context Engineering        → prompt vs context engineering, memory architecture,
#                                   context compression, KV cache, agent context lifecycle
# M19 DSPy                       → signatures, modules, MIPROv2, self-optimizing RAG
# M20 Agents                     → ReAct, MCP, LangGraph, CrewAI, browser agents,
#                                   A2A, guardrails, observability, evaluation
# M21 RL                         → PPO, GRPO, DAPO, GSPO, CISPO, reward models,
#                                   RLHF vs RLVR, policy gradient, DeepSeek-R1 training
 
# Cross-cutting rules:
# - "learning path / where to start"     → M1 → M2 → M3 in order
# - "production / deployment / serving"  → M9 (quantization) + M20 (agents)
# - "fine-tuning vs RAG"                 → M9 + M17 + M18
# - "multimodal / vision + language"     → M13 + M14 + M17 (multi-modal RAG)
# - "reasoning models / test-time RL"    → M9 (reasoning) + M21 (GRPO/DAPO)
# """


# # ── Expert-guided tree search ────────────────────────────────────────────────

# def llm_tree_search_with_expert(
#     query: str,
#     tree: list,
#     expert_rules: str,
#     model: str = "gpt-4o"
# ) -> dict:
#     """
#     Same as llm_tree_search() but with domain expert rules injected.
#     The LLM uses these rules to guide its reasoning.
#     """
    
#     def compress(nodes):
#         out = []
#         for n in nodes:
#             entry = {"node_id": n["node_id"], "title": n["title"],
#                      "page": n.get("page_index", "?"),
#                      "summary": n.get("text", "")[:150]}
#             if n.get("nodes"):
#                 entry["children"] = compress(n["nodes"])
#             out.append(entry)
#         return out

#     prompt = f"""You are a domain expert analyzing a document.
# Find all node IDs that most likely contain the answer to the query.
# Use the expert routing rules below to guide your reasoning.

# Query: {query}

# Document Tree:
# {json.dumps(compress(tree), indent=2)}

# Expert Routing Rules (follow these carefully):
# {expert_rules}

# Reply ONLY in this JSON format:
# {{
#   "thinking": "<your reasoning, referencing the expert rules>",
#   "node_list": ["node_id1", "node_id2"]
# }}"""

#     response = openai_client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"}
#     )
#     return json.loads(response.choices[0].message.content)


# # ── Test expert-guided retrieval ─────────────────────────────────────────────
# query = "Details of the modern llm finetuning?"

# print(f"🔍 Query: {query}\n")

# # Without expert rules
# print("── Without Expert Rules ──")
# basic   = llm_tree_search(query, pageindex_tree)
# print("Nodes:", basic.get("node_list"))

# print()

# # With expert rules
# print("── With Expert Rules ──")
# guided  = llm_tree_search_with_expert(query, pageindex_tree, FINANCIAL_EXPERT_RULES)
# print("Nodes:", guided.get("node_list"))
# print("Reasoning:", guided.get("thinking", "")[:300])


# # ── Full expert-guided RAG ───────────────────────────────────────────────────

# def expert_rag(query: str, tree: list, rules: str) -> str:
#     """Expert-guided end-to-end RAG pipeline."""
#     result  = llm_tree_search_with_expert(query, tree, rules)
#     nodes   = find_nodes_by_ids(tree, result.get("node_list", []))
#     return generate_answer(query, nodes)

# # Run it
# answer = expert_rag(
#     query="Details of the syllabus of modern llm finetuning",
#     tree=pageindex_tree,
#     rules=FINANCIAL_EXPERT_RULES
# )
# print(answer)






# # ── Single question with Chat API ────────────────────────────────────────────
# # No OpenAI key needed — PageIndex runs the LLM internally

# question = "What are the key findings in this document?"

# response = pi_client.chat_completions(
#     messages=[{"role": "user", "content": question}],
#     doc_id=doc_id
# )

# answer = response["choices"][0]["message"]["content"]
# print("💬 Chat API Answer:")
# print(answer)


# # ── Multi-turn conversation ───────────────────────────────────────────────────
# # Keep the full message history for context across turns

# conversation_history = []

# def chat_with_doc(user_message: str, doc_id: str) -> str:
#     """Chat with a document, maintaining conversation history."""
#     global conversation_history
    
#     conversation_history.append({"role": "user", "content": user_message})
    
#     response = pi_client.chat_completions(
#         messages=conversation_history,
#         doc_id=doc_id
#     )
    
#     assistant_reply = response["choices"][0]["message"]["content"]
#     conversation_history.append({"role": "assistant", "content": assistant_reply})
    
#     return assistant_reply


# # Simulate a 3-turn conversation
# questions = [
#     "What were the main revenue sources last year?",
#     "How does that compare to the year before?",
#     "What factors drove that change?"
# ]

# for q in questions:
#     print(f"\n👤 User: {q}")
#     reply = chat_with_doc(q, doc_id)
#     print(f"🤖 Assistant: {reply[:400]}...")
#     print("-" * 55)