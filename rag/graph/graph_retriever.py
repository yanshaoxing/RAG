"""
图检索模块 —— 从 Kuzu 知识图谱中检索与查询相关的实体和关系。

策略：
  1. 用 LLM 从查询中提取关键实体名称
  2. 在 Kuzu 中模糊匹配实体，找到入口节点
  3. 从入口节点出发，按指定深度遍历邻居
  4. 返回相关三元组文本

v2 变化：支持 Schema-aware 多 Label 查询（不再是硬编码 Entity）
"""

import json
import logging
import re
from typing import Optional, List

from llama_index.core.indices.property_graph import PropertyGraphIndex

from rag import config

logger = logging.getLogger(__name__)


ENTITY_EXTRACT_FROM_QUERY_PROMPT = (
    '你是一个实体识别助手。请从以下用户问题中提取所有具名实体（人物、组织、地点、物品、概念等）。\n'
    '不要提取代词（我、你、他、她）或泛称（某人、那个人）。\n'
    '如果问题中没有具名实体，返回空列表。\n\n'
    '输出格式（严格 JSON 数组）：\n'
    '["实体1", "实体2", ...]\n\n'
    '用户问题：{query}\n'
    '实体列表（JSON）：'
)


def _try_parse_json_array(text: str) -> Optional[list]:
    """尝试解析 JSON 数组。"""
    try:
        import json_repair
        result = json_repair.repair_json(text, return_objects=True)
        if isinstance(result, list):
            return result
    except (ImportError, Exception):
        pass

    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


class GraphRetriever:
    """从 Kuzu 知识图谱中检索与查询相关的子图信息。"""

    def __init__(
        self,
        graph_index: Optional[PropertyGraphIndex] = None,
        llm=None,
        log_list: Optional[list] = None,
    ):
        self._graph_index = graph_index
        self._llm = llm
        self._log_list = log_list

    def _log(self, msg: str):
        if self._log_list is not None:
            self._log_list.append(msg)

    @property
    def is_available(self) -> bool:
        return self._graph_index is not None and self._llm is not None

    def retrieve(self, query: str) -> str:
        """从图中检索与查询相关的三元组，返回格式化文本。

        如果图不可用或未检索到结果，返回空字符串。
        """
        if not self.is_available:
            self._log("步骤 3.5：图检索 — 知识图谱不可用，跳过")
            return ""

        entities = self._extract_entities_from_query(query)
        if not entities:
            self._log("步骤 3.5：图检索 — 查询中未识别到具名实体，跳过")
            return ""

        self._log(f"步骤 3.5：图检索 — 识别到实体: {entities}")

        triples = self._search_graph(entities)
        if not triples:
            self._log("步骤 3.5：图检索 — 未找到相关三元组")
            return ""

        self._log(f"步骤 3.5：图检索 — 找到 {len(triples)} 条相关三元组")

        formatted = self._format_triples(triples)
        return formatted

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """使用 LLM 从查询中提取具名实体。"""
        prompt = ENTITY_EXTRACT_FROM_QUERY_PROMPT.format(query=query)
        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()
            entities = _try_parse_json_array(text)
            if entities:
                return [e.strip() for e in entities if isinstance(e, str) and len(e.strip()) >= 2]
        except Exception as e:
            logger.warning(f"查询实体提取失败: {e}")
        return []

    def _search_graph(self, entities: List[str]) -> List[dict]:
        """在 Kuzu 图中搜索匹配实体的相关三元组。

        v2 变化：不再硬编码 Entity Label，使用通用 MATCH 匹配所有节点类型。
        """
        try:
            graph_store = self._graph_index.property_graph_store
            triples = []

            for entity in entities[: config.GRAPH_RETRIEVAL_TOP_K]:
                try:
                    # 使用通用 MATCH（不指定节点 Label），适配 Schema 多类型；
                    # 实体名通过参数传递，避免含引号等特殊字符时注入/报错
                    query_str = (
                        f"MATCH (a)-[r]->(b) "
                        f"WHERE a.name CONTAINS $entity OR b.name CONTAINS $entity "
                        f"RETURN a.name AS subject, r.label AS predicate, b.name AS object, "
                        f"       r.description AS description, r.chunk_id AS chunk_id "
                        f"LIMIT {int(config.GRAPH_RETRIEVAL_MAX_TRIPLES)}"
                    )
                    result = graph_store.structured_query(query_str, param_map={"entity": entity})
                    if result:
                        for row in result:
                            triples.append({
                                "subject": row.get("subject", ""),
                                "predicate": row.get("predicate", ""),
                                "object": row.get("object", ""),
                                "description": row.get("description", ""),
                                "chunk_id": row.get("chunk_id", None),
                            })
                except Exception as e:
                    logger.debug(f"实体 '{entity}' 图查询失败: {e}")
                    continue

            seen = set()
            unique_triples = []
            for t in triples:
                key = (t["subject"], t["predicate"], t["object"])
                if key not in seen:
                    seen.add(key)
                    unique_triples.append(t)
                    if len(unique_triples) >= config.GRAPH_RETRIEVAL_MAX_TRIPLES:
                        break

            return unique_triples

        except Exception as e:
            logger.warning(f"图搜索失败: {e}")
            return []

    def _format_triples(self, triples: List[dict]) -> str:
        """将三元组列表格式化为可嵌入 Prompt 的文本。"""
        lines = []
        for t in triples:
            line = f"- {t['subject']} → {t['predicate']} → {t['object']}"
            desc = t.get("description", "")
            if desc:
                line += f"\n  描述：{desc}"
            cid = t.get("chunk_id")
            if cid is not None:
                line += f"  [chunk #{cid}]"
            lines.append(line)
        return "\n".join(lines)