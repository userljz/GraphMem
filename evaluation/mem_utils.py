# retrieval_graph_nx.py
# -*- coding: utf-8 -*-
import re
import math
import random
from typing import Dict, List, Tuple, Optional
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------- Parse utils ----------------
def _extract_between(text: str, start_tag: str, end_tag: str, default: str = "") -> str:
    pat = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL)
    m = pat.search(text or "")
    return (m.group(1).strip() if m else default).strip()

def strip_prompt_get_question(prompt_block: str) -> str:
    return _extract_between(prompt_block, "<|im_start|>user", "<|im_end|>", default="")

def extract_code(full_output: str) -> str:
    code_blocks = re.findall(r"<python>(.*?)</python>", full_output or "", flags=re.DOTALL)
    if code_blocks:
        return "\n\n".join([b.strip() for b in code_blocks if b.strip()])
    backticks = re.findall(r"```(?:[\w+-]+)?\n(.*?)```", full_output or "", flags=re.DOTALL)
    if backticks:
        return "\n\n".join([b.strip() for b in backticks if b.strip()])
    return ""

def extract_think_without_code(full_output: str) -> str:
    text = re.sub(r"<python>.*?</python>", "", full_output or "", flags=re.DOTALL)
    text = re.sub(r"```(?:[\w+-]+)?\n.*?```", "", text, flags=re.DOTALL)
    return text.strip()

def build_full_trajectory(question: str, full_output: str) -> str:
    parts = []
    if question: parts.append(question.strip())
    if full_output: parts.append(full_output.strip())
    return "\n\n".join(parts).strip()

# ---------------- Qwen Reranker ----------------
class QwenReranker:
    MODEL = "Qwen/Qwen3-Reranker-0.6B"
    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, device: Optional[str] = None, torch_dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.MODEL, padding_side="left")
        self.mdl = AutoModelForCausalLM.from_pretrained(self.MODEL, torch_dtype=torch_dtype).to(self.device).eval()
        self.id_yes = self.tok.convert_tokens_to_ids("yes")
        self.id_no  = self.tok.convert_tokens_to_ids("no")

    @staticmethod
    def _format(instruct: str, query: str, doc: str) -> str:
        return (
            f"<|im_start|>system\n{QwenReranker.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}{QwenReranker.SUFFIX}"
        )

    # @torch.no_grad()
    # def score_batch(self, query: str, docs: List[str], instruct: str) -> List[float]:
    #     texts = [self._format(instruct, query, d) for d in docs]
    #     toks  = self.tok(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(self.mdl.device)
    #     out   = self.mdl(**toks)
    #     last  = out.logits[:, -1, :]
    #     yes_l = last[:, self.id_yes]
    #     no_l  = last[:, self.id_no]
    #     probs = torch.softmax(torch.stack([no_l, yes_l], dim=1), dim=1)[:, 1]
    #     return probs.tolist()

    @torch.no_grad()
    def score_batch(
        self,
        query: str,
        docs: List[str],
        instruct: str,
        *,
        batch_size: int = 5,        # 你想要的最大 batch
        max_length: int = 1024,     # 强烈建议同步降低 max_length，避免被极长样本拖长 padding
        clear_cache_each_chunk: bool = True,
        empty_doc_fallback: str = "<empty>",
    ) -> List[float]:
        """对 docs 分批打分，返回 P(yes) 列表。"""
        scores: List[float] = []
        n = len(docs)
        for st in range(0, n, batch_size):
            ed = min(st + batch_size, n)
            # 组装这一小批的输入（为空则用占位，避免 tokenizer 报错）
            chunk_docs = [(d if (d is not None and d.strip() != "") else empty_doc_fallback)
                        for d in docs[st:ed]]
            texts = [self._format(instruct, query, d) for d in chunk_docs]

            # Tokenize：左填充 + 截断，限制 max_length 以控显存
            toks = self.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.mdl.device)

            # 前向
            out  = self.mdl(**toks)
            last = out.logits[:, -1, :]  # [B, vocab]
            yes_l = last[:, self.id_yes]
            no_l  = last[:, self.id_no]
            probs = torch.softmax(torch.stack([no_l, yes_l], dim=1), dim=1)[:, 1]

            # 收集并释放中间变量，降低缓存占用
            scores.extend(probs.detach().float().cpu().tolist())
            del toks, out, last, yes_l, no_l, probs
            if clear_cache_each_chunk and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores

# ---------------- Graph manager (NetworkX) ----------------
class RetrievalGraphNX:
    """
    节点：整数 id，从 0 递增
    边：MultiDiGraph 上区分三/四类关系（通过 key='etype' 标识），并存 weight
    同时维护四个并行列表（与 node id 对齐）：query_list / code_list / think_list / trajectory_list
    """
    INSTRUCT_QUERY = "Determine whether the two queries ask for the same or highly similar problem."
    INSTRUCT_CODE  = "Determine whether the two code snippets implement the same or highly similar logic."
    INSTRUCT_THINK = "Determine whether the two texts (excluding code) describe similar reasoning or answers."
    INSTRUCT_ANS   = "Judge whether the two full answers (reasoning + final answer) are similar."

    def __init__(self, use_prob_sampling: bool = False):
        self.G = nx.MultiDiGraph()  # 有向多重图（允许不同 etype 的平行边）
        self.use_prob_sampling = use_prob_sampling

        self.query_list: List[str] = []
        self.code_list:  List[str] = []
        self.think_list: List[str] = []
        self.trajectory_list: List[str] = []

    # ---- helpers ----
    def _add_node(self) -> int:
        nid = self.G.number_of_nodes()
        self.G.add_node(nid)
        return nid

    def _add_bi_edge(self, u: int, v: int, etype: str, weight: float):
        # 双向各加一条，记录 etype & weight
        self.G.add_edge(u, v, etype=etype, weight=float(weight))
        self.G.add_edge(v, u, etype=etype, weight=float(weight))

    # ---- 仅 query 节点（建 query_similar 边）----
    def add_query_node(self, query: str, ranker: QwenReranker, threshold: float = 0.0) -> int:
        nid = self._add_node()
        # 列表占位
        self.query_list.append(query)
        self.code_list.append("")
        self.think_list.append("")
        self.trajectory_list.append("")

        if nid > 0:
            hist_ids = list(range(nid))
            hist_query = self.query_list[:-1]
            sims = ranker.score_batch(query, hist_query, self.INSTRUCT_QUERY)
            for j, s in zip(hist_ids, sims):
                if s >= threshold:
                    self._add_bi_edge(nid, j, "query_similar", s)
        return nid

    # ---- 完整 trajectory 节点（建四类边）----
    def add_full_trajectory(self, traj: Dict, ranker: QwenReranker,
                            th_query=0.0, th_code=0.0, th_think=0.0, th_ans=0.0) -> int:
        prompt      = str(traj.get("Prompt", ""))
        full_output = str(traj.get("Full_output", ""))

        q = strip_prompt_get_question(prompt)
        c = extract_code(full_output)
        t = extract_think_without_code(full_output)
        a = build_full_trajectory(q, full_output)

        nid = self._add_node()
        self.query_list.append(q)
        self.code_list.append(c)
        self.think_list.append(t)
        self.trajectory_list.append(a)

        if nid > 0:
            hist = list(range(nid))

            sims_q = ranker.score_batch(q, self.query_list[:-1], self.INSTRUCT_QUERY)
            for j, s in zip(hist, sims_q):
                if s >= th_query: self._add_bi_edge(nid, j, "query_similar", s)

            if any(self.code_list[:-1]):
                sims_c = ranker.score_batch(c or "<empty>", [x or "<empty>" for x in self.code_list[:-1]], self.INSTRUCT_CODE)
                for j, s in zip(hist, sims_c):
                    if s >= th_code: self._add_bi_edge(nid, j, "code_similar", s)

            sims_t = ranker.score_batch(t or "<empty>", [x or "<empty>" for x in self.think_list[:-1]], self.INSTRUCT_THINK)
            for j, s in zip(hist, sims_t):
                if s >= th_think: self._add_bi_edge(nid, j, "think_similar", s)

            sims_a = ranker.score_batch(a, self.trajectory_list[:-1], self.INSTRUCT_ANS)
            for j, s in zip(hist, sims_a):
                if s >= th_ans: self._add_bi_edge(nid, j, "answer_similar", s)

        return nid

    def remove_latest_node(self) -> int | None:
        """
        回滚最新加入的节点（最后一个 node_id），并清理其所有边与并行列表项。
        仅用于“这次 trajectory 不想保留/本次 query 临时加入后决定放弃”的场景。
        返回被移除的 node_id；若当前无节点则返回 None。
        """
        if not self.query_list:
            return None

        last_id = len(self.query_list) - 1

        # 1) 从图里移除该节点（会自动移除相连的所有边）
        if self.G.has_node(last_id):
            self.G.remove_node(last_id)
        # 注意：我们设计里 node_id 与四个列表索引对齐；只回滚“最新节点”才能保证一致性

        # 2) 回滚四个并行列表
        self.query_list.pop()
        self.code_list.pop()
        self.think_list.pop()
        self.trajectory_list.pop()

        return last_id

    # ---- Top-k 或 按权重采样的一跳 / 二跳 ----
    def _neighbors_by_etype(self, nid: int, etype: str) -> List[Tuple[int, float]]:
        """
        返回从 nid 出发、etype 边的 (dst, weight) 列表
        """
        out = []
        for _, v, data in self.G.out_edges(nid, data=True):
            if data.get("etype") == etype:
                out.append((v, float(data.get("weight", 0.0))))
        return out

    @staticmethod
    def _topk(pairs: List[Tuple[int, float]], k: int) -> List[int]:
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in pairs[:k]]

    @staticmethod
    def _sample_by_weight(pairs: List[Tuple[int, float]], k: int) -> List[int]:
        if not pairs or k <= 0: return []
        ids, w = zip(*pairs)
        total = sum(w)
        if total <= 0:
            # 无权重或全零，退化为均匀采样
            return random.sample(list(ids), k=min(k, len(ids)))
        probs = [x/total for x in w]
        # 允许重复就用 random.choices；不允许重复写个简单去重
        picked = []
        for _ in range(min(k, len(ids))):
            choice = random.choices(ids, weights=probs, k=1)[0]
            picked.append(choice)
            # 去重：把已选 id 的权重置 0 并重归一
            idx = ids.index(choice)
            w = list(w); w[idx] = 0.0
            total = sum(w)
            if total <= 0: break
            probs = [x/total for x in w]
            ids = list(ids)
        return list(dict.fromkeys(picked))

    def pick_neighbors_k1k2(self, node_id: int, k1: int = 5, k2: int = 3) -> Tuple[List[int], List[int]]:
        # 1-hop：只看 query_similar
        q_pairs = self._neighbors_by_etype(node_id, "query_similar")
        if self.use_prob_sampling:
            first_ids = self._sample_by_weight(q_pairs, k=k1)
        else:
            first_ids = self._topk(q_pairs, k=k1)

        # 2-hop：从 code/think/answer 三类边各取 k2
        second_set = set()
        for et in ["code_similar", "think_similar", "answer_similar"]:
            pool = []
            for u in first_ids:
                pool.extend(self._neighbors_by_etype(u, et))
            # 去掉自己和一跳
            pool = [(v, w) for (v, w) in pool if v != node_id and v not in first_ids]
            if self.use_prob_sampling:
                chosen = self._sample_by_weight(pool, k=k2)
            else:
                chosen = self._topk(pool, k=k2)
            second_set.update(chosen)

        return first_ids, sorted(second_set)

    def build_fewshot_prefix(self, node_id: int, k1: int = 5, k2: int = 3, sep: str = "\n\n") -> str:
        first, second = self.pick_neighbors_k1k2(node_id, k1, k2)
        blocks = [self.trajectory_list[i] for i in first + second if self.trajectory_list[i]]
        return sep.join(blocks).strip()
