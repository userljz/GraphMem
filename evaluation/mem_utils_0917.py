# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lobpcg
import re
import torch
from typing import List, Optional


# =================================================================================
# PART 1: Utilities from 0917_old_memory.py (Text Extraction & Reranker)
# =================================================================================

def _extract_between(text: str, start_tag: str, end_tag: str, default: str = "") -> str:
    """Helper to extract text between two tags."""
    pat = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL)
    m = pat.search(text or "")
    return (m.group(1).strip() if m else default).strip()


def extract_query(prompt_block: str) -> str:
    """Extracts the user question from a prompt block."""
    return _extract_between(prompt_block, "<|im_start|>user", "<|im_end|>", default="")


def extract_think(prompt_block: str) -> str:
    return _extract_between(prompt_block, "<think>", "</think>", default="")

# def extract_solution(prompt_block: str) -> str:
#     return _extract_between(prompt_block, "<think>", "</think>", default="")


def extract_solution_without_think(full_output: str) -> str:
    """
    通过移除 <think>...</think> 代码块，从完整输出中提取解决方案部分。

    Args:
        full_output (str): 模型的完整输出字符串，可能包含一个 <think> 块。

    Returns:
        str: 移除了 <think>...</think> 块并清理了前后空白字符的剩余字符串。
    """
    if not full_output:
        return ""
    
    # 定义一个正则表达式，用于匹配从 <think> 开始到 </think> 结束的整个代码块。
    # - `.*?` 是一个“非贪婪”匹配，它会匹配到第一个出现的 `</think>` 就停止。
    # - `flags=re.DOTALL` 确保 `.` 可以匹配包括换行符在内的任意字符。
    think_pattern = r"<think>.*?</think>"
    
    # 使用 re.sub() 函数，将匹配到的 think_pattern 替换为空字符串("")，即实现删除效果。
    solution_part = re.sub(think_pattern, "", full_output, flags=re.DOTALL)
    
    # 使用 .strip() 清理可能残余在字符串前后的空白或换行符。
    return solution_part.strip()




class QwenReranker:
    """Qwen Reranker for semantic similarity scoring."""
    MODEL = "Qwen/Qwen3-Reranker-0.6B"
    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    def __init__(self, device: Optional[str] = None, torch_dtype=torch.bfloat16):
        # Lazy import to avoid dependency if not used
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing QwenReranker on device: {self.device}")
        self.tok = AutoTokenizer.from_pretrained(self.MODEL, padding_side="left")
        self.mdl = AutoModelForCausalLM.from_pretrained(self.MODEL, torch_dtype=torch_dtype).to(self.device).eval()
        self.id_yes = self.tok.convert_tokens_to_ids("yes")
        self.id_no = self.tok.convert_tokens_to_ids("no")
    
    @staticmethod
    def _format(instruct: str, query: str, doc: str) -> str:
        return (
            f"<|im_start|>system\n{QwenReranker.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}{QwenReranker.SUFFIX}"
        )
    
    @torch.no_grad()
    def score_batch(self, query: str, docs: List[str], instruct: str, batch_size: int = 8) -> List[float]:
        if not docs:
            return []
        scores: List[float] = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i: i + batch_size]
            texts = [self._format(instruct, query, d or "<empty>") for d in batch_docs]
            toks = self.tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(
                self.mdl.device)
            out = self.mdl(**toks)
            last = out.logits[:, -1, :]
            yes_l = last[:, self.id_yes]
            no_l = last[:, self.id_no]
            probs = torch.softmax(torch.stack([no_l, yes_l], dim=1), dim=1)[:, 1]
            scores.extend(probs.tolist())
        return scores


# =================================================================================
# PART 2: Modified KnowledgeGraph class from 0917_memory_example.py
# =================================================================================

class KnowledgeGraph:
    """
    Incrementally-growing Q–T–S KG with incremental spectral approximation.
    Modified to use QwenReranker for similarity-based node merging.
    """
    INSTRUCT_QUERY = "Determine whether the two queries ask for the same or highly similar problem."
    INSTRUCT_THINK = "Determine whether the two texts (excluding code) describe similar reasoning or answers."
    INSTRUCT_CODE = "Determine whether the two code snippets implement the same or highly similar logic."
    
    # ----------------------------- init ---------------------------------
    def __init__(self, ranker: QwenReranker, alpha: float = 0.7):
        """
        Initializes the Knowledge Graph.
        Args:
            ranker (QwenReranker): An initialized instance of the QwenReranker.
            alpha (float): The similarity threshold for merging nodes.
        """
        self.G = nx.DiGraph()
        self.node_types, self.contents = {}, {}
        self.node_counter = 0
        # cached Laplacian eigen-pairs (k ≤ 10)
        self.evals, self.evecs = None, None
        self.k_eig = 10
        
        # --- MODIFIED ---
        self.ranker = ranker
        self.alpha = alpha
        self.reranker_instructs = {
            'query': self.INSTRUCT_QUERY,
            'thought': self.INSTRUCT_THINK,
            'solution': self.INSTRUCT_CODE,  # 'solution' now stores code
        }
    
    # ------------------------- add / merge ------------------------------
    def _add_node(self, txt: str, typ: str, parent=None) -> str:
        nid = f"{typ}_{self.node_counter}"
        self.node_counter += 1
        self.G.add_node(nid)
        self.node_types[nid] = typ
        self.contents[nid] = txt
        if parent:
            self.G.add_edge(parent, nid, weight=1.0)
        return nid
    
    def _merge_similar(self, new_content: str, node_type: str) -> Optional[str]:
        """
        Finds a similar existing node using the QwenReranker. If similarity is above alpha,
        merges content and returns the existing node ID. Otherwise, returns None.
        """
        # --- MODIFIED ---
        if not new_content.strip():  # Do not merge empty content
            return None
        
        # 1. Collect historical content of the same type
        hist_nodes = [nid for nid, typ in self.node_types.items() if typ == node_type]
        if not hist_nodes:
            return None
        hist_contents = [self.contents[nid] for nid in hist_nodes]
        
        # 2. Score similarity with the reranker
        instruction = self.reranker_instructs[node_type]
        scores = self.ranker.score_batch(new_content, hist_contents, instruction)
        
        if not scores:
            return None
        
        # 3. Check if the most similar node exceeds the threshold
        max_score = max(scores)
        if max_score >= self.alpha:
            best_match_idx = scores.index(max_score)
            nid_to_merge = hist_nodes[best_match_idx]
            
            # Optional: update content of the merged node
            # self.contents[nid_to_merge] += f"\n--- MERGED ---\n{new_content}"
            print(f"Merging {node_type} with {nid_to_merge} (score: {max_score:.2f} >= {self.alpha})")
            return nid_to_merge
        
        return None
    

    def update_graph(self, prompt: str, full_output: str):
        # 1. Extract Query, Thought, and Solution(Code) from inputs
        q_txt = extract_query(prompt)
        t_txt = extract_think(full_output)
        s_txt = extract_solution_without_think(full_output)

        if not q_txt:
            print("Warning: Could not extract question from prompt. Skipping graph update.")
            return None

        # 2. Merge or add the new query node
        merged_qid = self._merge_similar(q_txt, 'query')
        qid = merged_qid if merged_qid else self._add_node(q_txt, 'query')
        print(f"[QUERY] {'merged→' + merged_qid if merged_qid else 'new'}  →  {qid}")

        # 3. Merge or add the new thought node, connected to the query
        merged_tid = self._merge_similar(t_txt, 'thought')
        tid = merged_tid if merged_tid else self._add_node(t_txt, 'thought', parent=qid)
        print(f"[THINK] {'merged→' + merged_tid if merged_tid else 'new'}   →  {tid}")

        # 4. Merge or add the new solution(code) node, connected to the thought
        sid = None
        merged_sid = self._merge_similar(s_txt, 'solution')
        if s_txt:
            sid = merged_sid if merged_sid else self._add_node(s_txt, 'solution', parent=tid)
            print(f"[CODE ] {'merged→' + merged_sid if merged_sid else 'new'}   →  {sid}")
        else:
            print("[CODE ] (empty) — skipped")

        # 5. Update spectral components
        self._incremental_spectral()
        print(f"[GRAPH] updated. main query: {qid} | total nodes: {len(self.G.nodes)}\n")

        # 新增：把关键信息返回，方便主程序打印或做断言
        return {
            "qid": qid,
            "tid": tid,
            "sid": sid,
            "merged": {
                "query": bool(merged_qid),
                "thought": bool(merged_tid),
                "solution": bool(merged_sid),
            }
        }


    # def update_graph(self, prompt: str, full_output: str):
    #     """
    #     Adds a new trajectory (from prompt and full_output) to the graph,
    #     merging nodes if they are semantically similar to existing ones.
    #     """
    #     # --- MODIFIED ---
    #     # 1. Extract Query, Thought, and Solution(Code) from inputs
    #     q_txt = extract_query(prompt)
    #     t_txt = extract_think(full_output)
    #     s_txt = extract_solution_without_think(full_output)
        
    #     if not q_txt:
    #         print("Warning: Could not extract question from prompt. Skipping graph update.")
    #         return None
        
    #     # 2. Merge or add the new query node
    #     merged_qid = self._merge_similar(q_txt, 'query')
    #     qid = merged_qid if merged_qid else self._add_node(q_txt, 'query')
        
    #     # 3. Merge or add the new thought node, connected to the query
    #     merged_tid = self._merge_similar(t_txt, 'thought')
    #     tid = merged_tid if merged_tid else self._add_node(t_txt, 'thought', parent=qid)
        
    #     # 4. Merge or add the new solution(code) node, connected to the thought
    #     merged_sid = self._merge_similar(s_txt, 'solution')
    #     # A thought might not have a corresponding code solution
    #     if s_txt:
    #         sid = merged_sid if merged_sid else self._add_node(s_txt, 'solution', parent=tid)
        
    #     # 5. Update spectral components
    #     self._incremental_spectral()
    #     print(f"Updated graph with query: '{q_txt[:50]}...'. Main node: {qid}. Total nodes: {len(self.G.nodes)}\n")
    #     return qid
    
    # --------------------- incremental spectral (UNCHANGED) -------------------------
    def _incremental_spectral(self):
        n = self.G.number_of_nodes()
        if n <= 1:
            self.evals, self.evecs = None, None
            return
        A = nx.to_scipy_sparse_array(self.G, weight='weight', dtype=float).tocsr()
        deg = np.array(A.sum(axis=1)).flatten()
        L = diags(deg) - A
        k = min(self.k_eig, n - 1)
        if self.evecs is None or self.evecs.shape[0] != n:
            X = np.random.randn(n, k)
        else:
            X = np.zeros((n, k))
            # Pad existing eigenvectors if graph has grown
            prev_n = self.evecs.shape[0]
            X[:prev_n, :min(k, self.evecs.shape[1])] = self.evecs[:, :min(k, self.evecs.shape[1])]
            X += 1e-3 * np.random.randn(*X.shape)  # small noise for stability
        vals, vecs = lobpcg(L, X, largest=False, tol=1e-4, maxiter=60)
        idx = np.argsort(vals)
        self.evals = vals[idx]
        self.evecs = vecs[:, idx]
    
    # ----------------------- LPF / HPF utils (UNCHANGED) ----------------------------
    @staticmethod
    def _norm_adj(subG: nx.DiGraph) -> np.ndarray:
        A = nx.to_scipy_sparse_array(subG, weight='weight', dtype=float)
        deg = np.array(A.sum(axis=1)).flatten()
        # Avoid division by zero for isolated nodes
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sqrt_deg = 1.0 / np.sqrt(np.clip(deg, 1e-9, None))
            inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0
        D_inv_sqrt = diags(inv_sqrt_deg)
        P = D_inv_sqrt @ A @ D_inv_sqrt
        return P.toarray()
    
    @staticmethod
    def _low_pass(s, P, order=4):
        n = P.shape[0]
        I = np.eye(n)
        acc = s.copy()
        Pk = I.copy()
        for k in range(1, order + 1):
            Pk = Pk @ P
            acc += (1 / 2 ** k) * (Pk @ s)
        return acc
    
    @staticmethod
    def _high_pass(s, P, order=4):
        return s - KnowledgeGraph._low_pass(s, P, order)
    
    @staticmethod
    def _mmr(scores, nodes, lam=0.5, top_k=5):
        if not nodes: return []
        sel_indices, cand_indices = [], list(range(len(nodes)))
        while cand_indices and len(sel_indices) < top_k:
            mmr_scores = []
            for c_idx in cand_indices:
                # Similarity to query
                sim_to_query = lam * scores[c_idx]
                # Dissimilarity to already selected items
                max_sim_to_selected = 0
                if sel_indices:
                    # Using node index difference as a proxy for diversity
                    sims_to_selected = [1 / (1 + abs(c_idx - s_idx)) for s_idx in sel_indices]
                    max_sim_to_selected = max(sims_to_selected)
                
                mmr_scores.append(sim_to_query - (1 - lam) * max_sim_to_selected)
            
            best_cand_local_idx = int(np.argmax(mmr_scores))
            best_cand_global_idx = cand_indices[best_cand_local_idx]
            
            sel_indices.append(best_cand_global_idx)
            cand_indices.remove(best_cand_global_idx)
        
        return [nodes[i] for i in sel_indices]
    
    # ------------------------- k-hop subgraph (UNCHANGED) ---------------------------
    def _khop_sub(self, src, h=5):
        if src not in self.G: return nx.DiGraph()
        vis = {src}
        frontier = {src}
        for _ in range(h):
            if not frontier: break
            nxt = set()
            for v in frontier:
                nxt.update(self.G.successors(v))
                nxt.update(self.G.predecessors(v))
            frontier = nxt - vis
            vis.update(nxt)
        return self.G.subgraph(vis).copy()
    
    # ----------------------------- query (UNCHANGED) --------------------------------
    def find_nodes(self, new_q_txt, k1_queries, k2=10, k3=5):
        # 1) 临时插入新 Query 并连接到 k1_queries
        q_new = "temp_query_node_for_find"  # Use a unique temp name
        self.node_counter += 1  # Temporarily increment
        self.G.add_node(q_new)
        self.node_types[q_new] = 'query'
        self.contents[q_new] = new_q_txt
        
        temp_edges = []
        for q in k1_queries:
            if q in self.G.nodes:
                self.G.add_edge(q_new, q, weight=1.0)
                self.G.add_edge(q, q_new, weight=1.0)
                temp_edges.extend([(q_new, q), (q, q_new)])
        
        # 2) 取 5-hop 子图
        subG = self._khop_sub(q_new, 5)
        nodes = list(subG.nodes)
        if len(nodes) <= 1:
            self._cleanup(q_new, temp_edges)
            return list(k1_queries), [], []
        
        # 3) LPF 找 k2 个功能相似 queries
        P = self._norm_adj(subG)
        idx = {n: i for i, n in enumerate(nodes)}
        s = np.zeros((len(nodes), 1))
        s[idx[q_new]] = 1
        lp = self._low_pass(s, P)
        q_indices = [i for i, n in enumerate(nodes) if self.node_types.get(n) == 'query' and n != q_new]
        
        k2_queries = []
        if q_indices:
            q_scores = lp[q_indices, 0]
            # Get top k2 indices relative to the q_indices list
            top_k2_local_indices = np.argsort(q_scores)[-k2:][::-1]
            # Map back to original nodes list
            k2_queries = [nodes[q_indices[i]] for i in top_k2_local_indices]
        
        # 4) HPF + MMR 找 k3 个多样 thoughts/solutions
        hp = self._high_pass(s, P)
        ts_indices = [i for i, n in enumerate(nodes) if self.node_types.get(n) in {'thought', 'solution'}]
        diverse = []
        if ts_indices:
            ts_scores = hp[ts_indices, 0]
            ts_nodes = [nodes[i] for i in ts_indices]
            # Normalize scores for MMR if needed, though MMR is rank-based
            normalized_scores = (ts_scores - ts_scores.min()) / (ts_scores.max() - ts_scores.min() + 1e-9)
            diverse = self._mmr(normalized_scores, ts_nodes, top_k=k3)
        
        self._cleanup(q_new, temp_edges)
        return list(k1_queries), k2_queries, diverse
    
    def _cleanup(self, q_new, edges):
        if self.G.has_node(q_new):
            self.G.remove_node(q_new)
        if q_new in self.node_types: del self.node_types[q_new]
        if q_new in self.contents: del self.contents[q_new]
        self.node_counter -= 1
    
    def find_k1_queries(self, new_query_text: str, top_k: int = 3) -> List[str]:
        """
        Finds the top_k most similar query nodes from the graph based on a new query text.

        Args:
            new_query_text (str): The new user query.
            top_k (int): The number of similar queries to return.

        Returns:
            List[str]: A list of node IDs for the most similar queries.
        """
        # 1. Get all existing query nodes
        all_queries_nodes = [nid for nid, ntype in self.node_types.items() if ntype == 'query']
        
        if not all_queries_nodes:
            return []
        
        # Ensure we don't request more queries than exist
        k = min(top_k, len(all_queries_nodes))
        if k == 0:
            return []
        
        # 2. Get the content of all historical queries
        all_queries_content = [self.contents[nid] for nid in all_queries_nodes]
        
        # 3. Calculate similarity scores using the class's reranker instance
        scores = self.ranker.score_batch(new_query_text, all_queries_content, self.INSTRUCT_QUERY)
        
        # 4. Get the indices of the top k most similar queries
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # 5. Select and return the top k query node IDs
        k1_queries = [all_queries_nodes[i] for i in top_k_indices]
        
        return k1_queries
    
    def build_fewshot_prompt(self, retrieved_node_ids: list, max_examples: int = 3) -> str:
        """
        根据 find_nodes 返回的所有节点ID，重建因果链并构建Few-shot Prompt。

        Args:
            retrieved_node_ids (list): find_nodes返回的所有相关节点ID的列表 (k1+k2+k3)。
            max_examples (int): 最多构建多少个完整的问答示例。

        Returns:
            str: 格式化后的Few-shot Prompt字符串。
        """
        prompt_parts = []
        
        # 将所有检索到的节点ID放入一个集合中，以便快速查找
        retrieved_set = set(retrieved_node_ids)
        
        # 示例的入口点是所有被检索到的 query 节点
        entry_points = sorted([nid for nid in retrieved_set if self.node_types.get(nid) == 'query'])
        
        # 用一个集合来追踪哪些节点已经被用掉，避免重复构建
        used_nodes = set()
        example_count = 0
        
        for qid in entry_points:
            if qid in used_nodes or example_count >= max_examples:
                continue
            
            # --- 开始构建一个完整的示例 ---
            example_parts = []
            
            # 1. 添加Query部分
            q_content = self.contents.get(qid, "")
            example_parts.append(f"<|im_start|>user\n{q_content}<|im_end|>")
            example_parts.append(f"<|im_start|>assistant")
            used_nodes.add(qid)
            
            # 2. 寻找并添加所有与该Query相连、且同时也被检索到的Thought节点
            #    这是实现因果关系的关键
            child_thoughts = [
                tid for tid in sorted(self.G.successors(qid))
                if tid in retrieved_set and self.node_types.get(tid) == 'thought'
            ]
            
            all_thoughts_content = []
            all_solutions_content = []
            
            for tid in child_thoughts:
                if tid in used_nodes: continue
                
                all_thoughts_content.append(self.contents.get(tid, ""))
                used_nodes.add(tid)
                
                # 3. 为这个Thought寻找所有与它相连、且同时也被检索到的Solution节点
                child_solutions = [
                    sid for sid in sorted(self.G.successors(tid))
                    if sid in retrieved_set and self.node_types.get(sid) == 'solution'
                ]
                for sid in child_solutions:
                    if sid in used_nodes: continue
                    all_solutions_content.append(self.contents.get(sid, ""))
                    used_nodes.add(sid)
            
            # 4. 组装Assistant的回答部分，支持一对多
            if all_thoughts_content:
                full_thought_content = "\n".join(all_thoughts_content)
                example_parts.append(f"<think>{full_thought_content}</think>")
            
            if all_solutions_content:
                full_solution_content = "\n".join(all_solutions_content)
                example_parts.append(f"<python>\n{full_solution_content}\n</python>")
            
            # 将完整拼接好的一个示例加入列表
            prompt_parts.append("\n".join(example_parts))
            example_count += 1
        
        return "\n---\n".join(prompt_parts)

    def find_nodes_only_from_query(self, k1_queries: List[str]) -> tuple[List[str], List[str], List[str]]:
        """
        A simplified retrieval method that directly finds child nodes for a given list of query IDs.

        This function bypasses the complex LPF/HPF filtering. For each query in k1_queries,
        it finds its first associated 'thought' and that thought's first 'solution'.

        Args:
            k1_queries (List[str]): A list of query node IDs to start from.

        Returns:
            tuple[List[str], List[str], List[str]]: A tuple (k1, k2, k3) to maintain
            compatibility with build_fewshot_prompt.
            - k1: The original input list of k1_queries.
            - k2: Also the list of k1_queries, as these are the primary nodes found.
            - k3: A list of the found 'thought' and 'solution' node IDs.
        """
        # This list will store the retrieved thought and solution nodes for k3
        retrieved_ts_nodes = []

        # Iterate through each of the provided top query IDs
        for qid in k1_queries:
            # Find direct child nodes of type 'thought'
            child_thoughts = [
                tid for tid in self.G.successors(qid)
                if self.node_types.get(tid) == 'thought'
            ]

            # Per your request, even if there are multiple, only take the first one
            if child_thoughts:
                tid = child_thoughts[0]
                retrieved_ts_nodes.append(tid)

                # Now, find the first 'solution' child of that specific 'thought'
                child_solutions = [
                    sid for sid in self.G.successors(tid)
                    if self.node_types.get(sid) == 'solution'
                ]
                if child_solutions:
                    sid = child_solutions[0]
                    retrieved_ts_nodes.append(sid)

        # Return in the same format as find_nodes to ensure compatibility
        # k1 and k2 will both be the list of queries, k3 will be their children.
        # Use set() to ensure uniqueness of nodes in k3.
        return k1_queries, k1_queries, list(set(retrieved_ts_nodes))



# =================================================================================
# PART 3: New Demo Block
# =================================================================================
if __name__ == "__main__":
    # 1. Initialize the Reranker (requires GPU and transformers library)
    # Make sure you have run: pip install transformers torch accelerate
    try:
        reranker = QwenReranker()
    except Exception as e:
        print(f"Could not initialize QwenReranker. Please ensure you have a GPU and required libraries.")
        print(f"Error: {e}")
        
        
        # As a fallback for CPU testing, we can create a mock reranker
        class MockReranker:
            def score_batch(self, query, docs, instruct, **kwargs):
                print(
                    f"--- MOCK RERANKER: Comparing '{query[:20]}...' with {len(docs)} docs using instruct '{instruct.split()[1]}' ---")
                return [np.random.rand() for _ in docs]  # Returns random scores
        
        
        reranker = MockReranker()
    
    # 2. Initialize the Knowledge Graph with the reranker and a similarity threshold
    kg = KnowledgeGraph(ranker=reranker, alpha=0.6)  # Merge if score is >= 0.6
    
    # 3. Create dummy data in the new prompt/output format
    trajectories = [
        {
            "prompt": "<|im_start|>user\nHow do I plot a sine wave in Python?<|im_end|>",
            "Full_output": "<think> To solve the problem \\(10.0000198 \\cdot 5.9999985401 \\cdot 6.9999852\\) and find the result to the nearest whole number, we can use Python to perform the multiplication accurately. Let's calculate it step by step.\n\nFirst, we will multiply the three numbers together, and then we will round the result to the nearest whole number.\n </think><python>\n# Define the numbers\na = 10.0000198\nb = 5.9999985401\nc = 6.9999852\n\n# Perform the multiplication\nresult = a * b * c\n\n# Round the result to the nearest whole number\nnearest_whole_number = round(result)\nprint(nearest_whole_number)\n</python><result>\n420\n</result> <answer>The value of \\(10.0000198 \\cdot 5.9999985401 \\cdot 6.9999852\\) rounded to the nearest whole number is \\(\\boxed{420}\\).</answer>",
        },
        {
            "prompt": "<|im_start|>user\nCan you show me how to visualize a cosine function?<|im_end|>",
            "Full_output": "<think>  To determine the probability that the sum of the numbers on two fair 6-sided dice is 9, we can follow these steps:\n\n1. **Determine the total number of possible outcomes**: When rolling two 6-sided dice, there are \\(6 \\times 6 = 36\\) possible outcomes.\n2. **Determine the number of favorable outcomes**: We need to count the number of pairs \\((a, b)\\) where \\(a + b = 9\\). The possible pairs are \\((3, 6)\\), \\((4, 5)\\), \\((5, 4)\\), and \\((6, 3)\\). So, there are 4 favorable outcomes.\n3. **Calculate the probability**: The probability is the ratio of the number of favorable outcomes to the total number of possible outcomes. So, the probability is \\(\\frac{4}{36} = \\frac{1}{9}\\).\n\nLet's implement this in Python using sympy to ensure the result is accurate.\n\n </think><python>\nimport sympy as sp\n\n# Total number of possible outcomes when rolling two 6-sided dice\ntotal_outcomes = 6 * 6\n\n# Number of favorable outcomes where the sum is 9\nfavorable_outcomes = 4\n\n# Probability calculation\nprobability = sp.Rational(favorable_outcomes, total_outcomes)\nprint(probability)\n</python><result>\n1/9\n</result> <answer>The probability that the sum rolled is 9 when two fair 6-sided dice are rolled is \\(\\boxed{\\frac{1}{9}}\\).</answer>",
        },
        {
            "prompt": "<|im_start|>user\nHelp me create a scatter plot with random data.<|im_end|>",
            "Full_output": "<think>  To find the value of \\( x \\) that satisfies the equation \\( 6500 + x - 4500 = 3400 + 2000 \\), we can follow these steps:\n\n1. Simplify both sides of the equation.\n2. Isolate \\( x \\) on one side of the equation.\n\nLet's do this step-by-step.\n\nFirst, simplify both sides of the equation:\n\\[ 6500 - 4500 + x = 3400 + 2000 \\]\n\\[ 2000 + x = 5400 \\]\n\nNext, isolate \\( x \\) by subtracting 2000 from both sides of the equation:\n\\[ x = 5400 - 2000 \\]\n\\[ x = 3400 \\]\n\nSo, the value of \\( x \\) is \\( 3400 \\). Let's verify this by substituting \\( x = 3400 \\) back into the original equation to ensure it holds true.\n\nThe original equation is:\n\\[ 6500 + 3400 - 4500 = 3400 + 2000 \\]\n\nSubstituting \\( x = 3400 \\):\n\\[ 6500 + 3400 - 4500 = 3400 + 2000 \\]\n\\[ 9900 - 4500 = 5400 \\]\n\\[ 5400 = 5400 \\]\n\nSince both sides of the equation are equal, our solution is correct. The value of \\( x \\) is indeed \\( 3400 \\).\n\n</think><answer>The final answer is \\(\\boxed{3400}\\).</answer>",
        },
        {
            "prompt": "<|im_start|>user\nHow to draw a sine curve using Python?<|im_end|>",
            "Full_output": "<think> To determine the time when Bobbi's mother says they will be there in 7200 seconds, we need to convert this time into hours and then add it to the initial time of 2:30 p.m.\n\nHere's the step-by-step process:\n\n1. Convert 7200 seconds into minutes.\n2. Convert the total time from minutes into hours.\n3. Add the resulting hours to the initial time (2:30 p.m.).\n\nLet's do the calculations in Python:\n </think><python>\nfrom datetime import datetime, timedelta\n\n# Initial time in hours and minutes\ninitial_time = datetime.strptime(\"14:30\", \"%H:%M\")\n\n# Time in seconds after which they will arrive\ntime_in_seconds = 7200\n\n# Convert seconds to minutes\ntime_in_minutes = time_in_seconds // 60\n\n# Convert minutes to hours\ntime_in_hours = time_in_minutes // 60\n\n# Calculate the arrival time\narrival_time = initial_time + timedelta(hours=time_in_hours)\n\n# Format the arrival time as HH:MM\narrival_time_formatted = arrival_time.strftime(\"%H:%M\")\nprint(arrival_time_formatted)\n</python><result>\n16:30\n</result> <answer>The calculation shows that Bobbi's mother is correct, and they will arrive at their destination at \\(\\boxed{16:30}\\) p.m.</answer>",
        }
    ]
    
    for traj in trajectories:
        info = kg.update_graph(traj["prompt"], traj["Full_output"])  # 注意这里大小写
        if info:
            print(f"[CHOSEN] Q:{info['qid']}  T:{info['tid']}  S:{info['sid']}  "
                  f"(merged? {info['merged']})")

    # 5. Test the retrieval functionality
    all_queries_nodes = [nid for nid, ntype in kg.node_types.items() if ntype == 'query']

    if len(all_queries_nodes) >= 3:
        new_query_text = "How can I make a plot of a tan function?"
        k1_queries = kg.find_k1_queries(new_query_text, top_k=3)

        print("=" * 60)
        print(f"Testing retrieval with new query: '{new_query_text}'")
        print(f"Using context queries (k1): {k1_queries}")

        k1, k2, k3 = kg.find_nodes(new_query_text, k1_queries, k2=3, k3=2)

        print("\n--- Retrieval Results ---")
        print(f"Top 3 similar queries (external context): {k1}")
        print(f"Top 3 LPF similar queries from graph:    {k2}")
        print(f"Top 2 HPF diverse thoughts/solutions:    {k3}")

        all_retrieved_nodes = k1 + k2 + k3
        few_shot_prompt = kg.build_fewshot_prompt(all_retrieved_nodes, max_examples=2)

        # 关键：把 few-shot prompt 原样打出来
        print("\n--- FEW-SHOT PROMPT ---\n")
        print(few_shot_prompt)