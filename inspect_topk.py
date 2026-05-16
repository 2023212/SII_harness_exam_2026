"""
inspect_topk.py — 调试脚本

逻辑与 run.py 一致：先用训练集 update，再用测试集 predict。
区别：不计算准确率，而是把每条测试样本的 text 与检索到的 top-K 条
(text, label, similarity) 打印出来，用于人工检查 LSA 检索质量。

用法
----
  python inspect_topk.py
  python inspect_topk.py --top-k 10
  python inspect_topk.py --limit 20 --max-text-chars 200
  python inspect_topk.py --train data/train_dev.jsonl --dev data/test_dev.jsonl
"""

import argparse
import json
import sys

from solution import MyHarness


# Windows 重定向到文件时 sys.stdout 默认走 GBK，遇到 € / ™ / 表情等会抛 UnicodeEncodeError
# 统一切到 UTF-8，并用 replace 兜底避免任何罕见字符导致脚本崩掉
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _no_llm(_messages):
    # 检索流程不应该触达 LLM；触达即说明预测路径走错了
    raise RuntimeError("call_llm should not be invoked in retrieval-only debug mode")


def _approx_count_tokens(text: str) -> int:
    # 我们不走 LLM 路径，token 计数无实际意义，给个占位让构造器有东西可存
    return len(text)


def _approx_count_messages_tokens(messages) -> int:
    return sum(_approx_count_tokens(m.get("content", "")) for m in messages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train_dev.jsonl")
    parser.add_argument("--dev",   default="data/test_dev.jsonl")
    parser.add_argument("--top-k", type=int, default=5, help="检索深度（详情打印 + 严格不匹配统计）")
    parser.add_argument("--label-check-k", type=int, default=30,
                        help="更宽召回深度；按 label 去重后检查 true_label 是否被覆盖")
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 条测试样本，0 表示全部")
    parser.add_argument("--max-text-chars", type=int, default=160,
                        help="打印时单条 text 的最大字符数，0 表示不截断")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()

    train = load_jsonl(args.train)
    dev = load_jsonl(args.dev)
    if args.limit > 0:
        dev = dev[:args.limit]

    print(f"Train: {len(train)} 条 | Dev: {len(dev)} 条 | top_k={args.top_k}\n")

    harness = MyHarness(
        _no_llm,
        _approx_count_tokens,
        _approx_count_messages_tokens,
        args.max_prompt_tokens,
    )

    # 与 run.py 一致：按顺序灌入训练流（update 阶段只缓存，不做 SVD）
    print("[1/3] update 阶段（缓存训练样本与分词结果）...")
    for item in train:
        harness.update(item["text"], item["label"])

    # 提前触发懒拟合，把 SVD 开销从第一条 predict 上挪开，便于阅读
    print("[2/3] 首次拟合 LSA（构造 TF-IDF + 截断 SVD）...")
    harness._ensure_fitted()
    print(f"      vocab={len(harness._vocab_idx)}  doc-emb shape={harness.lsa_vectors.shape}\n")

    print("[3/3] 在测试集上检索 top-K，打印结果\n")

    def _trim(s: str) -> str:
        if args.max_text_chars > 0 and len(s) > args.max_text_chars:
            return s[:args.max_text_chars].rstrip() + "..."
        return s

    def _print_entry(idx, query_text, true_label, results, sim_range=None):
        extra = f"  sim_range={sim_range:+.4f}" if sim_range is not None else ""
        print("=" * 80)
        print(f"[Dev #{idx}] true_label={true_label}{extra}")
        print(f"  query : {_trim(query_text)}")
        print(f"  ----- top-{args.top_k} retrieved -----")
        if not results:
            print("  (无结果)")
            print()
            return
        for rank, (text, label, sim) in enumerate(results, 1):
            match_mark = "  <- label matches query" if label == true_label else ""
            print(f"  #{rank}  sim={sim:+.4f}  label={label}{match_mark}")
            print(f"        text: {_trim(text)}")
        print()

    # 收集 top-K 内所有 label 都 != true_label 的"硬错"条目
    mismatches: list[dict] = []
    # 收集 top-{wide_k} 去重 label 后仍不含 true_label 的"超硬错"条目
    label_miss_wide: list[dict] = []
    wide_k = max(args.top_k, args.label_check_k)

    for i, item in enumerate(dev):
        query_text = item["text"]
        true_label = item.get("label", "<no-gold>")
        # 一次 predict 拉到 wide_k；前 args.top_k 切片给详情/旧统计用，整段拿去做 label 覆盖检查
        wide_results = harness.predict(query_text, top_k=wide_k)
        results = wide_results[:args.top_k]

        _print_entry(i, query_text, true_label, results)

        if results and all(label != true_label for _, label, _ in results):
            sims = [s for _, _, s in results]
            mismatches.append({
                "idx": i,
                "query": query_text,
                "true_label": true_label,
                "results": results,
                "sim_range": max(sims) - min(sims),
            })

        # 按 sim 降序去重 label，看 true_label 是否落在更宽召回的去重集合里
        if wide_results:
            seen: set = set()
            wide_labels_dedup: list[str] = []
            for _, lab, _ in wide_results:
                if lab not in seen:
                    seen.add(lab)
                    wide_labels_dedup.append(lab)
            if true_label not in seen:
                label_miss_wide.append({
                    "idx": i,
                    "query": query_text,
                    "true_label": true_label,
                    "wide_labels": wide_labels_dedup,
                })

    # ========== 汇总：top-K 内 label 全部不匹配 true_label 的条目 ==========
    print("=" * 80)
    print(f"汇总：top-{args.top_k} 内 label 全部 != true_label 的测试条目数 = {len(mismatches)}")
    print("=" * 80 + "\n")

    if not mismatches:
        print("(无)\n")
    elif len(mismatches) < 10:
        print("(不足 10 条，全部列出，未按 sim 极差排序)\n")
        for m in mismatches:
            _print_entry(m["idx"], m["query"], m["true_label"], m["results"], m["sim_range"])
    else:
        by_range = sorted(mismatches, key=lambda m: m["sim_range"], reverse=True)
        print("--- sim 极差最大的 5 条（top-K 内相似度跨度最大） ---\n")
        for m in by_range[:5]:
            _print_entry(m["idx"], m["query"], m["true_label"], m["results"], m["sim_range"])
        print("--- sim 极差最小的 5 条（top-K 内相似度普遍接近） ---\n")
        for m in reversed(by_range[-5:]):
            _print_entry(m["idx"], m["query"], m["true_label"], m["results"], m["sim_range"])

    # ========== 新增汇总：top-{wide_k} 去重 label 仍未覆盖 true_label 的条目 ==========
    print("=" * 80)
    print(f"汇总：top-{wide_k}（按 label 去重）仍不含 true_label 的测试条目数 = {len(label_miss_wide)}")
    print(f"      （这些样本无论 LLM rerank 候选放多宽都救不了，需要 query expansion 或扩训练集）")
    print("=" * 80 + "\n")

    for m in label_miss_wide:
        labels_preview = m["wide_labels"][:8]
        suffix = f"  ...(+{len(m['wide_labels']) - 8})" if len(m["wide_labels"]) > 8 else ""
        print(f"  [Dev #{m['idx']}] true={m['true_label']}")
        print(f"       query : {_trim(m['query'])}")
        print(f"       retrieved labels (unique, sim-desc): {', '.join(labels_preview)}{suffix}")
        print()


if __name__ == "__main__":
    main()
