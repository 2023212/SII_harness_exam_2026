"""
solution.py — 考生唯一需要提交的文件

规则
----
1. 只能修改 MyHarness 类内部；其余部分不可改动。考生可以先行查看 harness_base.py 以了解可用接口和调用约定。
2. 只允许 import Python 标准库（re, math, random, json, collections 等）、numpy
   以及 harness_base（已提供）。
3. 禁止 import 其他第三方库（openai, sklearn, torch …）。
4. 禁止通过任何途径读写磁盘文件。
5. call_llm 每次调用的 prompt token 数若超过 max_prompt_tokens，
   会被自动截断至预算上限后再发送，
   可用 count_tokens（计算单条消息的 token 数） 和 count_messages_tokens（计算消息列表的总 token 数）预先控制 prompt 长度。
6. predict() 只接收 text，任何绕过接口获取 label 的行为将导致得分归零。
"""

import json
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np

from harness_base import Harness


# ============================================================
# 考生实现区（考生只能修改 MyHarness 类里的内容）
# ============================================================
class MyHarness(Harness):
    LSA_DIM = 1024            # 截断 SVD 的目标维度上限（实际维度被 min(N, |V|, LSA_DIM) 限制）
    _MIN_NGRAM_LEN = 4        # 词长 >= 此值时附加 char-ngram 特征
    _CHAR_NGRAMS = (3, 4)     # 字符 n-gram 的 n 取值
    _RE_WORD = re.compile(r"[a-z0-9]+|[一-鿿]")

    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        super().__init__(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)
        # 训练样本只在 update 时缓存，真正的 LSA 拟合延迟到首次查询
        self.raw_texts: list[str] = []
        self.raw_labels: list[str] = []
        self._token_lists: list[list[str]] = []
        # 拟合后才填充：词表、IDF、截断 SVD 的右奇异基矩阵
        self._vocab_idx: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._S_k: np.ndarray = np.zeros(0, dtype=np.float32)
        self._Vt_k: np.ndarray = np.zeros((0, 0), dtype=np.float32)  # shape (k, |V|)
        # 训练文档在潜在语义空间的坐标，shape (N, k)，与 lsa_labels 平行
        self.lsa_vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.lsa_labels: list[str] = []
        # (text, label, vector) 三元组列表，predict 检索时按下标取用
        self.memory_entries: list[tuple[str, str, np.ndarray]] = []
        self._fit_dirty: bool = False  # 自上次 fit 以来是否新增过样本

    # ---------- 分词：词 + 词 bigram + 拉丁词 char n-gram ----------
    def _tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text).lower()
        words = self._RE_WORD.findall(text)
        tokens: list[str] = list(words)
        # 词序列 bigram；连续的 CJK 单字也借此形成"中|国"二元组
        tokens.extend(f"{a}|{b}" for a, b in zip(words, words[1:]))
        # 拉丁词的字符 n-gram（CJK 单字长度=1，自然不进入分支）
        for w in words:
            if len(w) >= self._MIN_NGRAM_LEN:
                wb = f"<{w}>"
                for n in self._CHAR_NGRAMS:
                    tokens.extend(wb[i:i + n] for i in range(len(wb) - n + 1))
        return tokens

    # ---------- 流式只缓存，把 SVD 推迟到首次需要 ----------
    def update(self, text: str, label: str) -> None:
        self.raw_texts.append(text)
        self.raw_labels.append(label)
        self._token_lists.append(self._tokenize(text))
        self._fit_dirty = True
        super().update(text, label)

    def _ensure_fitted(self) -> None:
        """首次或在 update 之后调用，懒拟合 LSA 基矩阵 + 训练向量。"""
        if not self._fit_dirty:
            return
        self._fit_lsa()
        self._fit_dirty = False

    def _fit_lsa(self) -> None:
        N = len(self.raw_texts)
        if N == 0:
            return
        # 词表与文档频率
        df: Counter = Counter()
        for toks in self._token_lists:
            df.update(set(toks))
        vocab = sorted(df)
        self._vocab_idx = {t: i for i, t in enumerate(vocab)}
        V = len(vocab)
        if V == 0:
            return
        # 平滑 IDF：log((N+1)/(df+1)) + 1
        idf = np.empty(V, dtype=np.float32)
        for t, i in self._vocab_idx.items():
            idf[i] = np.log((N + 1.0) / (df[t] + 1.0)) + 1.0
        self._idf = idf
        # TF-IDF 矩阵 M (N × V)，sublinear TF + 行 L2 归一化
        M = np.zeros((N, V), dtype=np.float32)
        for i, toks in enumerate(self._token_lists):
            for t, c in Counter(toks).items():
                j = self._vocab_idx.get(t)
                if j is not None:
                    M[i, j] = np.log1p(c)
            M[i] *= idf
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        M /= norms
        # 截断 SVD：M ≈ U_k Σ_k V_k^T
        k = max(1, min(self.LSA_DIM, N, V))
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        U_k = U[:, :k].astype(np.float32)
        self._S_k = S[:k].astype(np.float32)
        self._Vt_k = Vt[:k, :].astype(np.float32)
        # 训练文档在潜在空间的坐标 = U_k * Σ_k，再行 L2 归一化以便余弦
        doc_embs = U_k * self._S_k[None, :]
        n2 = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        n2[n2 == 0] = 1.0
        self.lsa_vectors = (doc_embs / n2).astype(np.float32)
        self.lsa_labels = list(self.raw_labels)
        # 把 (text, label, vector) 打包为元组，向量直接复用矩阵的行视图，零额外拷贝
        self.memory_entries = [
            (self.raw_texts[i], self.raw_labels[i], self.lsa_vectors[i])
            for i in range(N)
        ]

    def _embed(self, text: str) -> np.ndarray:
        """把任意文本投影到训练好的 LSA 子空间，返回 L2 归一化后的向量。"""
        self._ensure_fitted()
        V = len(self._vocab_idx)
        if V == 0 or self._Vt_k.size == 0:
            return np.zeros(0, dtype=np.float32)
        q = np.zeros(V, dtype=np.float32)
        for t, c in Counter(self._tokenize(text)).items():
            j = self._vocab_idx.get(t)
            if j is not None:
                q[j] = np.log1p(c)
        q *= self._idf
        n = float(np.linalg.norm(q))
        if n > 0.0:
            q /= n
        # fold-in：q @ V_k = q @ Vt_k.T，得到潜在空间的查询向量
        q_emb = q @ self._Vt_k.T
        n2 = float(np.linalg.norm(q_emb))
        if n2 > 0.0:
            q_emb /= n2
        return q_emb.astype(np.float32)

    def predict(self, text: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        # 与 update 同套向量化（_embed 内部会触发懒拟合），再与训练向量库做余弦
        # 返回前 top_k 条 (text, label, similarity)，top_k 设为参数方便后续调优
        q = self._embed(text)
        if q.size == 0 or not self.memory_entries:
            return []
        # lsa_vectors 与 q 都已 L2 归一化，矩阵乘积即逐条余弦相似度
        sims = self.lsa_vectors @ q
        k = min(top_k, len(self.memory_entries))
        if k <= 0:
            return []
        # argpartition 拿到无序的 top-k 索引，再按相似度降序排
        part_idx = np.argpartition(-sims, k - 1)[:k]
        order = part_idx[np.argsort(-sims[part_idx])]
        return [
            (self.memory_entries[i][0], self.memory_entries[i][1], float(sims[i]))
            for i in order
        ]

    # 如需要，可以设计其他辅助方法
