"""
抽取模块 —— 使用 LLM 从 chunk 中抽取实体和关系，应用规则过滤，计算置信度。

与旧版的关键区别：
  1. 使用 json_repair 替代正则解析 JSON
  2. 每个实体/关系携带置信度分数
  3. 规则从 rules.yaml 加载，可配置
  4. Schema 驱动的类型解析
"""

import json
import logging
import os
from typing import Optional

from rag.utils.json_parse import parse_json_obj

from .models import ChunkResult, Entity, Relation
from .schema import Schema

logger = logging.getLogger(__name__)


def _load_rules() -> dict:
    """加载规则配置：基础规则（rag/graph/rules.json，语料无关）+ 语料补充规则合并。

    语料补充规则（corpora/<slug>/graph_rules.json，可缺省）按键合并：
    列表取并集（保序去重），标量覆盖基础值。
    """
    rules_path = os.path.join(os.path.dirname(__file__), "rules.json")
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception:
        logger.warning("无法加载 rules.json，使用空规则集")
        rules = {}

    try:
        from rag import config
        if os.path.exists(config.GRAPH_RULES_PATH):
            with open(config.GRAPH_RULES_PATH, "r", encoding="utf-8") as f:
                corpus_rules = json.load(f)
            for key, value in corpus_rules.items():
                base = rules.get(key)
                if isinstance(value, list) and isinstance(base, list):
                    rules[key] = list(dict.fromkeys(base + value))
                else:
                    rules[key] = value
    except Exception as e:
        logger.warning("无法加载语料补充规则 %s：%s", config.GRAPH_RULES_PATH, e)

    return rules


class Extractor:
    """从文本 chunk 中抽取实体和关系。"""

    def __init__(
        self,
        llm,
        schema: Schema,
        extract_prompt: str,
        model_name: str = "",
    ):
        self._llm = llm
        self._schema = schema
        self._extract_prompt = extract_prompt
        self._model_name = model_name
        self._rules = _load_rules()

    def extract(self, chunk_text: str, chunk_id: int) -> Optional[ChunkResult]:
        """从单个 chunk 中抽取实体和关系。

        Returns:
            ChunkResult 如果抽取成功，None 如果失败或无有效内容。
        """
        if not chunk_text or len(chunk_text.strip()) < 20:
            return None

        # 调用 LLM 抽取
        prompt = self._extract_prompt.format(chunk_text=chunk_text[:3000])
        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()
        except Exception as e:
            logger.warning(f"LLM 抽取异常 chunk #{chunk_id}: {e}")
            return None

        # 解析 JSON（parse_json_obj 保证返回 dict 或 None，不会因 LLM 输出数组而抛 AttributeError）
        data = parse_json_obj(text)
        if not data:
            logger.debug(f"chunk #{chunk_id}: 无法解析 LLM 返回的 JSON")
            return None

        raw_entities = data.get("entities", [])
        raw_relations = data.get("relations", [])
        if not isinstance(raw_entities, list):
            raw_entities = []
        if not isinstance(raw_relations, list):
            raw_relations = []

        # 处理实体
        entities, valid_names = self._process_entities(raw_entities, chunk_id)

        # 处理关系
        relations = self._process_relations(raw_relations, valid_names, entities, chunk_id, chunk_text)

        # 注意：只有实体、没有关系的 chunk 也返回结果（实体描述仍有价值），
        # 由调用方决定如何处理，不再一律视为失败重抽
        return ChunkResult(
            chunk_id=chunk_id,
            entities=entities,
            relations=relations,
            raw_text=chunk_text[:200],
        )

    def _process_entities(
        self, raw_entities: list[dict], chunk_id: int
    ) -> tuple[list[Entity], set[str]]:
        """处理原始实体列表，应用规则过滤，计算置信度。"""
        entities: list[Entity] = []
        valid_names: set[str] = set()

        min_name_len = self._rules.get("min_entity_name_length", 2)
        pronoun_blacklist = set(self._rules.get("pronoun_blacklist", []))
        generic_blacklist = set(self._rules.get("generic_blacklist", []))

        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "") or "").strip()

            # 规则过滤
            if len(name) < min_name_len:
                continue
            if name in pronoun_blacklist:
                continue
            if name in generic_blacklist:
                continue

            raw_type = ent.get("type", "未知")
            resolved_type = self._schema.resolve_type(raw_type)
            desc = ent.get("description", "").strip()

            # 计算置信度
            confidence = self._compute_entity_confidence(name, raw_type, resolved_type, desc)

            entity = Entity(
                name=name,
                type=resolved_type,
                description=desc,
                confidence=confidence,
                chunk_id=chunk_id,
            )
            entities.append(entity)
            valid_names.add(name)

        return entities, valid_names

    def _process_relations(
        self,
        raw_relations: list[dict],
        valid_names: set[str],
        entities: list[Entity],
        chunk_id: int,
        chunk_text: str,
    ) -> list[Relation]:
        """处理原始关系列表，应用规则过滤，计算置信度。"""
        entity_type_map = {e.name: e.type for e in entities}

        min_pred_len = self._rules.get("min_predicate_length", 2)
        min_desc_len = self._rules.get("min_description_length", 5)
        predicate_blacklist = set(self._rules.get("trivial_predicate_blacklist", []))
        known_male = set(self._rules.get("known_male_characters", []))
        female_keywords = self._rules.get("female_only_keywords", [])
        possession_preds = set(self._rules.get("possession_predicates", []))

        relations: list[Relation] = []
        seen_keys: set[tuple] = set()

        for rel in raw_relations:
            if not isinstance(rel, dict):
                continue
            subj = str(rel.get("subject", "") or "").strip()
            obj = str(rel.get("object", "") or "").strip()
            pred = str(rel.get("predicate", "") or "").strip()

            # 实体名称有效性
            if subj not in valid_names or obj not in valid_names:
                continue

            # 谓词过滤
            if len(pred) < min_pred_len:
                continue
            if pred in predicate_blacklist:
                continue

            # 性别校验
            if not self._check_gender(subj, pred, obj, known_male, female_keywords, possession_preds):
                continue

            # 去重（同一 chunk 内）
            key = (subj, pred, obj)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # 规范化谓词
            normalized_pred = self._schema.normalize_predicate(pred)

            desc = rel.get("description", "").strip()
            subj_type = entity_type_map.get(subj, "Entity")
            obj_type = entity_type_map.get(obj, "Entity")

            # 计算置信度
            confidence = self._compute_relation_confidence(pred, desc, subj_type, obj_type)

            relations.append(Relation(
                subject=subj,
                predicate=normalized_pred,
                object=obj,
                subject_type=subj_type,
                object_type=obj_type,
                description=desc,
                chunk_id=chunk_id,
                source_text=chunk_text[:200],
                confidence=confidence,
                extract_model=self._model_name,
            ))

        return relations

    def _compute_entity_confidence(self, name: str, raw_type: str, resolved_type: str, desc: str) -> float:
        """计算实体置信度（0.0 ~ 1.0）。"""
        confidence = 0.5
        if len(desc) >= 10:
            confidence += 0.2
        if resolved_type != "Entity":
            confidence += 0.2
        if len(name) >= 3:
            confidence += 0.1
        return min(confidence, 1.0)

    def _compute_relation_confidence(self, pred: str, desc: str, subj_type: str, obj_type: str) -> float:
        """计算关系置信度（0.0 ~ 1.0）。"""
        confidence = 0.5
        if len(desc) >= 10:
            confidence += 0.2
        if self._schema.is_known_predicate(pred):
            confidence += 0.15
        if subj_type != "Entity" and obj_type != "Entity":
            confidence += 0.15
        return min(confidence, 1.0)

    @staticmethod
    def _check_gender(
        subj: str, pred: str, obj: str,
        known_male: set[str], female_keywords: list[str], possession_preds: set[str],
    ) -> bool:
        """性别一致性校验。"""
        if subj not in known_male:
            return True
        if pred not in possession_preds:
            return True
        for keyword in female_keywords:
            if keyword in obj:
                logger.debug(f"性别校验过滤: {subj}(男) → {pred} → {obj}(含'{keyword}')")
                return False
        return True