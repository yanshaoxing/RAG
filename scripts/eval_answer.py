"""回答评测 —— 跑完整问答（检索 + 合成），再用"异模型"LLM-as-judge 打分。

judge 用与回答模型不同的模型（config.ALIYUN_VALIDATE_MODEL，沿用 create_validate_llm 的
交叉校验思路），从 忠实度 / 引用 / 完整度 三维打 1~5 分，并统计命中的标准答案要点数。
judge 提示词在 rag/prompts.py::EVAL_JUDGE_TEMPLATE_STR。

用法（仓库根目录）：
  python scripts/eval_answer.py                          # 默认语料，默认 qa 集
  python scripts/eval_answer.py -c WuLingChaShi --limit 3
  python scripts/eval_answer.py --config-override RERANK_ENABLED=false
  python scripts/eval_answer.py --save eval/baseline_answer.json

注意：本脚本每题要花一次回答 LLM + 一次 judge LLM，比 eval_retrieval 贵。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_common import (  # noqa: E402
    aggregate_answer,
    format_gold_points,
    load_qa,
    parse_overrides,
)
from scripts.eval_retrieval import _apply_overrides, _config_snapshot  # noqa: E402


def _collect_answer(response) -> str:
    """把查询结果（可能是流式）收敛成完整字符串。"""
    if hasattr(response, "response_gen"):
        return "".join(response.response_gen)
    return str(response)


def _build_judge():
    """构造 judge LLM：优先 create_validate_llm（异模型）；被开关关掉时直接用校验模型兜底。"""
    from rag import config
    from rag.llm.factory import _create_aliyun_llm, create_validate_llm

    llm = create_validate_llm()
    if llm is not None:
        return llm
    # GRAPH_VALIDATE_ENABLED=False 时 create_validate_llm 返回 None；评测仍需一个异模型 judge
    return _create_aliyun_llm(model_name=config.ALIYUN_VALIDATE_MODEL,
                              temperature=config.GRAPH_VALIDATE_LLM_TEMPERATURE)


def _judge_one(judge_llm, question: str, gold_points: list[str], answer: str) -> dict:
    from rag import prompts
    from rag.utils.json_parse import parse_json_obj

    prompt = prompts.EVAL_JUDGE_TEMPLATE_STR.format(
        question=question,
        num_points=len(gold_points),
        gold_points=format_gold_points(gold_points),
        answer=answer,
    )
    from llama_index.core.llms import ChatMessage, MessageRole
    resp = judge_llm.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
    obj = parse_json_obj(resp.message.content) or {}

    def clamp5(v) -> int:
        try:
            return max(1, min(5, int(round(float(v)))))
        except (TypeError, ValueError):
            return 1

    hit = obj.get("hit_points", 0)
    try:
        hit = max(0, min(len(gold_points), int(hit)))
    except (TypeError, ValueError):
        hit = 0
    return {
        "faithfulness": clamp5(obj.get("faithfulness")),
        "citation": clamp5(obj.get("citation")),
        "completeness": clamp5(obj.get("completeness")),
        "hit_points": hit,
        "num_points": len(gold_points),
        "reason": str(obj.get("reason", ""))[:200],
    }


def run(corpus_slug: str, qa_path: str, limit: int | None) -> dict:
    from rag import corpus
    from rag.engine.bootstrap import build_query_engine, init_settings
    from rag.logging_utils import capture_pipeline_logs

    profile = corpus.load_profile(corpus_slug)
    qa = load_qa(qa_path)
    if limit:
        qa = qa[:limit]
    print(f"语料《{profile.title}》（{corpus_slug}）：{len(qa)} 题")

    init_settings()
    with capture_pipeline_logs():
        engine = build_query_engine(corpus_slug)
    judge_llm = _build_judge()

    per_q = []
    for row in qa:
        with capture_pipeline_logs():
            response = engine.query(row["question"])
            answer = _collect_answer(response)
        scores = _judge_one(judge_llm, row["question"], row["gold_answer_points"], answer)
        per_q.append({
            "id": row.get("id"),
            "type": row.get("type", "unknown"),
            "question": row["question"],
            "answer": answer,
            "scores": scores,
        })
        print(f"  [{row.get('id')} {row.get('type')}] "
              f"忠实={scores['faithfulness']} 引用={scores['citation']} "
              f"完整={scores['completeness']} 命中{scores['hit_points']}/{scores['num_points']}"
              f"  {row['question'][:22]}")

    return {
        "corpus": corpus_slug,
        "qa_path": qa_path,
        "n": len(qa),
        "config": _config_snapshot(),
        "aggregate": aggregate_answer(per_q),
        "per_question": per_q,
    }


def _print_report(result: dict) -> None:
    agg = result["aggregate"]
    print("\n" + "=" * 60)
    print("回答评测汇总（忠实度 / 引用 / 完整度，1~5）")
    print("=" * 60)

    def line(name: str, s: dict) -> None:
        if s.get("n", 0) == 0:
            return
        phr = s["point_hit_rate"]
        phr_s = f"{phr:.2f}" if phr is not None else "-"
        print(f"  {name:<10} n={s['n']:<3} 忠实={s['faithfulness']:.2f}  引用={s['citation']:.2f}  "
              f"完整={s['completeness']:.2f}  要点命中率={phr_s}")

    line("总体", agg["overall"])
    for t, s in agg["by_type"].items():
        line(t, s)


def main() -> int:
    p = argparse.ArgumentParser(description="RAG 回答评测（LLM-as-judge）")
    p.add_argument("-c", "--corpus", default=None, help="语料 slug（默认取激活语料）")
    p.add_argument("--qa", default=None, help="qa.jsonl 路径（默认 eval/<slug>/qa.jsonl）")
    p.add_argument("--config-override", action="append", default=[], metavar="KEY=VALUE",
                   help="临时覆盖 rag.config（可重复），须在构建引擎前生效")
    p.add_argument("--save", default=None, metavar="PATH", help="把完整结果写成 JSON")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟）")
    args = p.parse_args()

    from rag import corpus
    slug = args.corpus or corpus.get_active_slug()
    qa_path = args.qa or f"eval/{slug}/qa.jsonl"
    if not os.path.exists(qa_path):
        print(f"找不到 qa 集：{qa_path}", file=sys.stderr)
        return 1

    try:
        overrides = parse_overrides(args.config_override)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if overrides:
        _apply_overrides(overrides)

    result = run(slug, qa_path, args.limit)
    _print_report(result)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已保存：{args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
