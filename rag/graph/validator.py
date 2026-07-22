"""
校验模块 —— 使用 LLM 对低置信度三元组进行判别式校验。

与旧版的关键区别：
  1. 仅校验置信度低于阈值的 triplet（节省 token）
  2. 使用 json_repair 解析 JSON
  3. 返回含 validate_model 标记的 Relation
"""

import logging

from rag.utils.json_parse import parse_json_obj, coerce_index_set

from .models import Relation

logger = logging.getLogger(__name__)


class Validator:
    """对低置信度三元组进行 LLM 判别式校验。"""

    def __init__(
        self,
        validate_llm,
        validate_prompt: str,
        model_name: str = "",
        confidence_threshold: float = 0.7,
        enabled: bool = True,
    ):
        self._llm = validate_llm
        self._validate_prompt = validate_prompt
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._enabled = enabled

    def validate(self, relations: list[Relation], chunk_text: str) -> list[Relation]:
        """校验关系列表。

        只有置信度低于阈值的关系才会被送 LLM 校验。
        高置信度关系直接通过。

        Args:
            relations: 待校验的关系列表
            chunk_text: 原文片段

        Returns:
            通过校验（或修正后）的关系列表
        """
        if not self._enabled or not relations:
            return relations

        # 分类：低置信度 → 校验，高置信度 → 直接通过
        high_conf = [r for r in relations if r.confidence >= self._confidence_threshold]
        low_conf = [r for r in relations if r.confidence < self._confidence_threshold]

        if not low_conf:
            return relations

        # 注意：即使只有一条低置信度关系也送 LLM 校验 ——
        # 此前"单条直接放行"导致单关系 chunk 永远不被校验
        logger.info(
            f"  🔍 校验: {len(high_conf)} 条高置信度直接通过, "
            f"{len(low_conf)} 条低置信度送 LLM 校验"
        )

        # 构建校验 Prompt
        triples_text = "\n".join(
            f"{i}. {r.subject} → {r.predicate} → {r.object}"
            f"{'  [' + r.description + ']' if r.description else ''}"
            for i, r in enumerate(low_conf)
        )

        prompt = self._validate_prompt.format(
            chunk_text=chunk_text[:3000],
            triples_text=triples_text,
        )

        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()
        except Exception as e:
            logger.warning(f"LLM 校验异常: {e}，保留全部低置信度关系")
            for r in low_conf:
                r.validated = True
                r.validate_model = self._model_name
            return high_conf + low_conf

        data = parse_json_obj(text)
        if not data:
            logger.debug("校验 LLM 未返回有效 JSON，保留全部低置信度关系")
            for r in low_conf:
                r.validated = True
                r.validate_model = self._model_name
            return high_conf + low_conf

        # 下标统一规整为 int（LLM 可能返回 ["0","2"] 等字符串下标，
        # 否则匹配全部失败 → 该 chunk 的低置信度关系被整体误删）
        valid_indices = coerce_index_set(data.get("valid", []))
        invalid_indices = sorted(coerce_index_set(data.get("invalid", [])))
        corrected_list = data.get("corrected", [])
        if not isinstance(corrected_list, list):
            corrected_list = []
        reasons = data.get("reasons", {})
        if not isinstance(reasons, dict):
            reasons = {}

        # 防御：若 LLM 未返回任何有效判定，保留全部低置信度关系
        if not valid_indices and not invalid_indices and not corrected_list:
            logger.debug("校验 LLM 返回的判定为空，保留全部低置信度关系")
            for r in low_conf:
                r.validated = True
                r.validate_model = self._model_name
            return high_conf + low_conf

        # 日志：被过滤的
        if invalid_indices:
            logger.info(f"  🔍 LLM 校验过滤掉 {len(invalid_indices)}/{len(low_conf)} 条:")
            for i in invalid_indices:
                if 0 <= i < len(low_conf):
                    r = low_conf[i]
                    reason = reasons.get(str(i), "") or reasons.get(i, "")
                    logger.info(
                        f"     ❌ [{i}] {r.subject} → {r.predicate} → {r.object}"
                        f"{'  — ' + reason if reason else ''}"
                    )

        # 收集通过校验的
        validated_low = []
        for i, r in enumerate(low_conf):
            if i in valid_indices:
                r.validated = True
                r.validate_model = self._model_name
                validated_low.append(r)

        # 收集修正的
        for corr in corrected_list:
            if not isinstance(corr, dict):
                continue
            try:
                idx = int(corr.get("index", -1))
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(low_conf):
                orig = low_conf[idx]
                corrected = Relation(
                    subject=corr.get("subject", orig.subject),
                    predicate=corr.get("predicate", orig.predicate),
                    object=corr.get("object", orig.object),
                    subject_type=orig.subject_type,
                    object_type=orig.object_type,
                    description=corr.get("description", orig.description),
                    chunk_id=orig.chunk_id,
                    source_text=orig.source_text,
                    confidence=orig.confidence + 0.1,  # 修正后略微提升置信度
                    validated=True,
                    extract_model=orig.extract_model,
                    validate_model=self._model_name,
                )
                validated_low.append(corrected)
                logger.info(
                    f"     🔧 [{idx}] {orig.subject} → {orig.predicate} → {orig.object}"
                    f"  →  {corrected.subject} → {corrected.predicate} → {corrected.object}"
                )

        # 合并高置信度 + 校验后的低置信度，去重
        result = high_conf + validated_low
        seen: set[tuple] = set()
        unique_result = []
        for r in result:
            if r.triple_key not in seen:
                seen.add(r.triple_key)
                unique_result.append(r)

        return unique_result