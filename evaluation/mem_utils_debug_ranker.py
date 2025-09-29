# -*- coding: utf-8 -*-
from __future__ import annotations
import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lobpcg
import re
import torch


from typing import List, Tuple, Dict, Optional


# =================================================================================
# PART 1: Utilities from 0917_old_memory.py (Text Extraction & Reranker)
# =================================================================================

# def _extract_between(text: str, start_tag: str, end_tag: str, default: str = "") -> str:
#     """Helper to extract text between two tags."""
#     pat = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL)
#     m = pat.search(text or "")
#     return (m.group(1).strip() if m else default).strip()

def _extract_between(text: str, start_tag: str, end_tag: str, default: str = "") -> str:
    """Helper to extract text between two tags (returns the LAST occurrence)."""
    if not text:
        return default.strip()
    
    # Escape tags and build pattern
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    else:
        return default.strip()



def extract_query(prompt_block: str) -> str:
    """Extracts the user question from a prompt block."""
    return _extract_between(prompt_block, "<|im_start|>user", "<|im_end|>", default="")


# def extract_think(prompt_block: str) -> str:
#     return _extract_between(prompt_block, "<think>", "</think>", default="")

# def extract_solution(prompt_block: str) -> str:
#     return _extract_between(prompt_block, "<think>", "</think>", default="")


def extract_solution(full_output: str) -> str:
    
    if not full_output:
        return ""
  
    # result_pattern = r"<result>.*?</result>"
    # answer_pattern = r"<answer>.*?</answer>"
    
    # solution_part = re.sub(result_pattern, "", full_output, flags=re.DOTALL)
    # solution_part = re.sub(answer_pattern, "", solution_part, flags=re.DOTALL)
    # solution_part = full_output

    think_part = _extract_between(full_output, "<python>", "</python>", default="")
    
    return think_part.strip()


class QwenReranker:
    """Qwen Reranker for semantic similarity scoring."""
    MODEL = "Qwen/Qwen3-Reranker-4B"
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
    简化版 Query–Answer 知识图：
      - 不做节点合并：每条 trajectory = {query_i, answer_i}
      - 边：
          * Q–Q：相似度加权边（非负）
          * A–A：相似度加权边（非负）
          * Q–A（配对）：权重=1（无权也可，但为便于LPF取1）
      - 检索：仅低通滤波（LPF），直接在全图上做，不取子图
      - 依赖：
          * ranker: 需提供 score_batch(query, candidates, instruction) -> List[float]
    """

    # INSTRUCT_QUERY = "Determine whether the two queries ask for the same or highly similar problem."
    # INSTRUCT_ANSWER = "Determine whether the two solutions describe similar reasoning or answers."

    INSTRUCT_QUERY = """
    Rank similarity between two queries for few-shot selection.
    Prioritize same task intent, output format, domain/difficulty, and constraints (APIs, languages,
    precision, units, limits). Reward semantic equivalence beyond paraphrasing; penalize mismatched
    outputs or incompatible constraints.
    """

    INSTRUCT_ANSWER = """
    Rank similarity between two solutions for few-shot selection.
    Prioritize same algorithmic idea, complexity class, data structures/APIs, edge-case handling,
    and I/O contracts. Ignore naming/style. Penalize different algorithms or incompatible outputs.
    """


    def __init__(self, ranker, lpf_order: int = 4):
        """
        Args:
            ranker: 例如 QwenReranker，需实现 score_batch
            lpf_order: 低通滤波多项式阶数（越大扩散更远，计算更慢）
        """
        self.G: nx.Graph = nx.Graph()               # 使用无向图，便于相似度建模
        self.node_types: Dict[str, str] = {}        # nid -> {'query','answer'}
        self.contents: Dict[str, str] = {}          # nid -> text
        self.pair: Dict[str, str] = {}              # 成对映射：qid <-> aid
        self.q_count = 0
        self.a_count = 0

        self.ranker = ranker
        self.lpf_order = lpf_order

    # --------------------------- 基础工具 ---------------------------

    def _new_id(self, typ: str) -> str:
        if typ == 'query':
            nid = f"query_{self.q_count}"
            self.q_count += 1
            return nid
        elif typ == 'answer':
            nid = f"answer_{self.a_count}"
            self.a_count += 1
            return nid
        else:
            raise ValueError("typ must be 'query' or 'answer'")

    def _add_node(self, text: str, typ: str) -> str:
        nid = self._new_id(typ)
        self.G.add_node(nid)
        self.node_types[nid] = typ
        self.contents[nid] = text
        return nid

    @staticmethod
    def _safe_weight(x: float) -> float:
        # 若模型可能产生负分或NaN，这里裁剪为非负并处理异常
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return 0.0
        return float(max(0.0, x))

    # --------------------------- 图构建：新增一条轨迹 ---------------------------

    def add_trajectory(self, query_text: str, answer_text: str) -> Tuple[str, str]:
        """
        新增一条 Q–A 轨迹，不做合并。
        自动添加：
          - Q–A（配对，权重=1）
          - Q 与所有历史 Q 的相似度边（Q–Q）
          - A 与所有历史 A 的相似度边（A–A）
        """
        query_text = extract_query(query_text)
        answer_text = extract_solution(answer_text)

        qid = self._add_node(query_text, 'query')
        aid = self._add_node(answer_text, 'answer')

        # 配对 Q–A：权重=1
        self.G.add_edge(qid, aid, weight=1.0)
        self.pair[qid] = aid
        self.pair[aid] = qid

        # 连接 Q–Q
        old_qs = [n for n, t in self.node_types.items() if t == 'query' and n != qid]
        if old_qs:
            old_q_texts = [self.contents[n] for n in old_qs]
            scores = self.ranker.score_batch(query_text, old_q_texts, self.INSTRUCT_QUERY) or []
            for nid_old, s in zip(old_qs, scores):
                w = self._safe_weight(s)
                if w > 0:
                    self.G.add_edge(qid, nid_old, weight=w)

        # 连接 A–A
        old_as = [n for n, t in self.node_types.items() if t == 'answer' and n != aid]
        if old_as:
            old_a_texts = [self.contents[n] for n in old_as]
            scores = self.ranker.score_batch(answer_text, old_a_texts, self.INSTRUCT_ANSWER) or []
            for nid_old, s in zip(old_as, scores):
                w = self._safe_weight(s)
                if w > 0:
                    self.G.add_edge(aid, nid_old, weight=w)

        return qid, aid

    # --------------------------- 检索：仅低通滤波（全图） ---------------------------

    @staticmethod
    def _norm_adj(G: nx.Graph, nodelist: List[str]):
        """
        对称归一化邻接矩阵 P = D^{-1/2} A D^{-1/2}
        使用 nodelist 固定行列与节点顺序对应关系，避免稀疏阵与索引错位。
        """
        A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight='weight', dtype=float).tocsr()
        deg = np.array(A.sum(axis=1)).ravel()
        inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
        Dm12 = diags(inv_sqrt)
        P = Dm12 @ A @ Dm12
        return P

    @staticmethod
    def _low_pass(s: np.ndarray, P, order: int = 4) -> np.ndarray:
        """
        多项式低通（Neumann-like 累加）：acc = s + 1/2 P s + 1/4 P^2 s + ...
        s: (n, 1)
        """
        n = s.shape[0]
        I = np.eye(n)
        acc = s.copy()
        Pk = I.copy()
        for k in range(1, order + 1):
            Pk = Pk @ P
            acc += (1 / (2 ** k)) * (Pk @ s)
        return acc

    def _temp_connect_query(self, q_tmp: str, query_text: str) -> List[Tuple[str, str]]:
        """
        将临时 query 与所有历史 query 建立 Q–Q 相似度边；返回已加边（用于回收）。
        """
        edges_added = []
        old_qs = [n for n, t in self.node_types.items() if t == 'query']
        if not old_qs:
            return edges_added
        texts = [self.contents[n] for n in old_qs]
        scores = self.ranker.score_batch(query_text, texts, self.INSTRUCT_QUERY) or []
        for nid_old, s in zip(old_qs, scores):
            w = self._safe_weight(s)
            if w > 0:
                self.G.add_edge(q_tmp, nid_old, weight=w)
                edges_added.append((q_tmp, nid_old))
        return edges_added

    def _cleanup_temp(self, q_tmp: str, edges: List[Tuple[str, str]]):
        for u, v in edges:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
        if self.G.has_node(q_tmp):
            self.G.remove_node(q_tmp)
        self.node_types.pop(q_tmp, None)
        self.contents.pop(q_tmp, None)

    def find_related_trajectories(
        self,
        new_query_text: str,
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        基于新 query 做一次 LPF 检索，返回 Top-K 相关轨迹 (qid, aid, score)。
        步骤：
          1) 将临时节点 q_tmp 加入全图，仅与所有历史 query 建 Q–Q 相似度边
          2) 在全图上构建 P，并在 q_tmp 位置打脉冲，进行 LPF
          3) 以每条轨迹得分 = score(q_i) + score(a_i) 排序取 Top-K
          4) 清理临时节点与边
        """
        # 1) 临时节点
        q_tmp = "_temp_query_node_"
        if q_tmp in self.G:
            self.G.remove_node(q_tmp)
        self.G.add_node(q_tmp)
        self.node_types[q_tmp] = 'query'
        self.contents[q_tmp] = new_query_text

        temp_edges = self._temp_connect_query(q_tmp, new_query_text)

        # 图规模检查
        if self.G.number_of_nodes() <= 1:
            self._cleanup_temp(q_tmp, temp_edges)
            return []

        # 2) 全图节点列表（固定顺序）
        nodes = list(self.G.nodes)
        idx = {n: i for i, n in enumerate(nodes)}

        # 3) P 与 LPF（在全图）
        P = self._norm_adj(self.G, nodelist=nodes)
        s = np.zeros((len(nodes), 1))
        s[idx[q_tmp]] = 1.0
        lp = self._low_pass(s, P, order=self.lpf_order).ravel()

        # 4) 轨迹联合得分：score(traj_i) = LPF(q_i) + LPF(a_i)
        traj_scores = []
        for qid in nodes:
            if self.node_types.get(qid) != 'query' or qid == q_tmp:
                continue
            aid = self.pair.get(qid)
            if aid is None:
                continue
            iq = idx.get(qid)
            ia = idx.get(aid)
            if iq is None or ia is None:
                continue
            score = float(lp[iq]) + float(lp[ia])
            traj_scores.append((qid, aid, score))

        traj_scores.sort(key=lambda x: x[2], reverse=True)
        top = traj_scores[:top_k]

        # 5) 清理
        self._cleanup_temp(q_tmp, temp_edges)
        return top

    # --------------------------- Few-shot 拼装（可选） ---------------------------

    def build_fewshot_prompt(
        self,
        existing_prompt,
        traj_list: List[Tuple[str, str, float]],
        max_examples: int = 3
    ) -> str:
        """
        根据 find_related_trajectories 的返回 (qid, aid, score) 构建 few-shot。
        """
        parts = []
        for qid, aid, score in traj_list[:max_examples]:
            if score > 0:
                q = self.contents.get(qid, "")
                a = self.contents.get(aid, "")
                parts.append(
                    f"<|im_start|>user\n{q}<|im_end|>\n"
                    f"<|im_start|>assistant\n{a}<|im_end|>"
                )

        fewshot_prefix = "\n\n".join(parts)

        system_prompt = _extract_between(existing_prompt, "<|im_start|>system", "<|im_end|>", default="")
        system_prompt = "<|im_start|>system" + "\n\n" + system_prompt + "\n" + "<|im_end|>"
        new_query_prompt = _extract_between(existing_prompt, "<|im_start|>user", "<|im_end|>", default="")
        new_query_prompt = "<|im_start|>user" + "\n\n" + new_query_prompt + "\n" + "<|im_end|>" + "\n" + "<|im_start|>assistant" + "\n"
        ret_prompt = system_prompt + "\n\n" + fewshot_prefix + "\n\n" + new_query_prompt

        return ret_prompt

# =================================================================================
# PART 3: New Demo Block
# =================================================================================
if __name__ == "__main__":
    import torch

    # 初始化 reranker
    rr = QwenReranker()

    # 定义 instruct / query / 文档
    instruct = "根据文档内容判断其是否满足查询要求；只能回答 yes 或 no。"
    query = "该文档是否支持七天无理由退货？"
    docs = [
        "本店支持七天无理由退货，商品不影响二次销售即可办理。",
        "您可在签收后7日内办理退货，无需说明理由。",
        "我们支持质量问题退货，需在签收后7日内联系客服。",
        "本店不支持无理由退货，除非存在严重质量问题。",
        "这是一份关于物流派送范围的说明。",
    ]

    # 批量打分
    scores = rr.score_batch(query=query, docs=docs, instruct=instruct, batch_size=2)

    print("\n========== 结果 ==========")
    for i, (d, s) in enumerate(zip(docs, scores)):
        preview = (d[:60] + "…") if len(d) > 60 else d
        print(f"[#{i}] score={s:.4f} | {preview}")

    # 验证相似说法是否打分更高
    print("\n========== 排序展示 ==========")
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    for doc, sc in ranked:
        preview = (doc[:80] + "…") if len(doc) > 80 else doc
        print(f"score={sc:.4f} | {preview}")


# ========== 结果 ==========
# [#0] score=0.9883 | 本店支持七天无理由退货，商品不影响二次销售即可办理。
# [#1] score=0.9727 | 您可在签收后7日内办理退货，无需说明理由。
# [#2] score=0.4375 | 我们支持质量问题退货，需在签收后7日内联系客服。
# [#3] score=0.1406 | 本店不支持无理由退货，除非存在严重质量问题。
# [#4] score=0.0002 | 这是一份关于物流派送范围的说明。

# ========== 排序展示 ==========
# score=0.9883 | 本店支持七天无理由退货，商品不影响二次销售即可办理。
# score=0.9727 | 您可在签收后7日内办理退货，无需说明理由。
# score=0.4375 | 我们支持质量问题退货，需在签收后7日内联系客服。
# score=0.1406 | 本店不支持无理由退货，除非存在严重质量问题。
# score=0.0002 | 这是一份关于物流派送范围的说明。