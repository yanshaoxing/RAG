"""
LLM 输出 JSON 解析工具 —— json_repair 优先，正则提取兜底。

此前同样的 "repair-then-regex" 逻辑在 extractor / validator /
canonicalizer / graph_retriever 中复制了 4 份，且未处理 json_repair
可能返回 list/str 等非预期类型的情况（曾导致 data.get() 抛 AttributeError
中断整个图构建）。统一收敛到此处，并保证返回类型。
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def _repair(text: str):
    """尝试 json_repair 解析，失败返回 None。"""
    try:
        import json_repair
        return json_repair.repair_json(text, return_objects=True)
    except ImportError:
        return None
    except Exception:
        return None


def parse_json_obj(text: str) -> dict | None:
    """从 LLM 输出中解析 JSON 对象。返回 dict，无法解析（或结果不是 dict）返回 None。"""
    if not text:
        return None

    result = _repair(text)
    if isinstance(result, dict):
        return result

    # 回退：正则提取最外层 {...}
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def parse_json_list(text: str) -> list | None:
    """从 LLM 输出中解析 JSON 数组。返回 list，无法解析（或结果不是 list）返回 None。"""
    if not text:
        return None

    result = _repair(text)
    if isinstance(result, list):
        return result

    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def coerce_index_set(values) -> set[int]:
    """把 LLM 返回的下标列表规整为 int 集合（LLM 可能返回 ["0","2"] 等字符串下标）。"""
    result: set[int] = set()
    if not isinstance(values, (list, tuple, set)):
        return result
    for v in values:
        try:
            result.add(int(v))
        except (ValueError, TypeError):
            continue
    return result
