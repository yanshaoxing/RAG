"""
Schema 管理模块 —— 管理实体类型、关系类型规范化、Schema 自动成长。

功能：
  1. 内置实体类型（Person, Organization, Location, ...）
  2. 未知类型自动发现，达到阈值后自动升级为 learned 类型
  3. 关系谓词规范化（"出生于" → "born_in"）
  4. 生成 Build Fingerprint，用于缓存失效检测
"""

import hashlib
import json
import logging
import os
from typing import Optional

from .models import SchemaTypeInfo

logger = logging.getLogger(__name__)

# 内置实体类型：这些类型直接对应 Kuzu 的 Node Label
BUILTIN_ENTITY_TYPES: dict[str, str] = {
    "人物": "Person",
    "组织": "Organization",
    "公司": "Organization",
    "机构": "Organization",
    "地点": "Location",
    "城市": "Location",
    "国家": "Location",
    "事件": "Event",
    "物品": "Item",
    "产品": "Item",
    "概念": "Concept",
    "称号": "Concept",
    "金钱数额": "Concept",
    "作品": "CreativeWork",
    "书籍": "CreativeWork",
    "音乐": "CreativeWork",
}

# 关系谓词规范化映射：中文谓词 → 英文标准谓词
RELATION_NORMALIZE_MAP: dict[str, str] = {
    "出生于": "born_in",
    "出生地点": "born_in",
    "出生在": "born_in",
    "生于": "born_in",
    "任职于": "works_at",
    "就职于": "works_at",
    "工作于": "works_at",
    "在...工作": "works_at",
    "供职于": "works_at",
    "创办": "founded",
    "创立": "founded",
    "创建": "founded",
    "成立": "founded",
    "爱慕": "loves",
    "相爱": "loves",
    "恋人": "loves",
    "配偶": "spouse_of",
    "结婚": "spouse_of",
    "嫁给": "spouse_of",
    "娶了": "spouse_of",
    "朋友": "friend_of",
    "好友": "friend_of",
    "知己": "friend_of",
    "合作": "collaborates_with",
    "合伙人": "collaborates_with",
    "冲突": "conflicts_with",
    "对立": "conflicts_with",
    "针对": "conflicts_with",
    "评价": "evaluates",
    "评论": "evaluates",
    "评价了": "evaluates",
    "赠送": "gives_to",
    "送给": "gives_to",
    "赠予": "gives_to",
    "借钱": "lends_to",
    "借款": "lends_to",
    "委托": "entrusts",
    "委托给": "entrusts",
    "属于": "belongs_to",
    "隶属": "belongs_to",
    "管辖": "governs",
    "管理": "manages",
    "领导": "leads",
    "指导": "mentors",
    "教导": "mentors",
    "策划": "plans",
    "设计": "designs",
    "投资": "invests_in",
    "收购": "acquires",
    "拥有": "owns",
    "持有": "owns",
    "参与": "participates_in",
    "加入": "joins",
    "离开": "leaves",
    "退出": "leaves",
    "居住": "lives_in",
    "住在": "lives_in",
    "留学": "studies_in",
    "学习": "studies_at",
    "毕业于": "graduated_from",
    "出版": "publishes",
    "写作": "writes",
    "创作": "creates",
    "演奏": "performs",
    "演唱": "performs",
}


class Schema:
    """管理知识图谱的 Schema 定义和自动成长。"""

    def __init__(
        self,
        builtin_types: Optional[dict[str, str]] = None,
        relation_map: Optional[dict[str, str]] = None,
        growth_threshold: int = 5,
    ):
        self._builtin: dict[str, str] = builtin_types or dict(BUILTIN_ENTITY_TYPES)
        self._relation_map: dict[str, str] = relation_map or dict(RELATION_NORMALIZE_MAP)
        self._growth_threshold = growth_threshold

        # 未知类型池：{type_name: count}
        self._unknown_types: dict[str, int] = {}

        # 已学习的类型：{type_name: kuzu_label}
        self._learned_types: dict[str, str] = {}

    # ---- Entity Type ----

    def resolve_type(self, raw_type: str) -> str:
        """将 LLM 抽取的原始类型转换为 Kuzu Label。

        优先级：builtin > learned > Entity（兜底）
        """
        if not raw_type:
            return "Entity"

        # 精确匹配 builtin
        if raw_type in self._builtin:
            return self._builtin[raw_type]

        # 精确匹配 learned
        if raw_type in self._learned_types:
            return self._learned_types[raw_type]

        # 模糊匹配 builtin（子串包含）
        for cn_type, kuzu_label in self._builtin.items():
            if cn_type in raw_type or raw_type in cn_type:
                return kuzu_label

        # 模糊匹配 learned
        for cn_type, kuzu_label in self._learned_types.items():
            if cn_type in raw_type or raw_type in cn_type:
                return kuzu_label

        # 未识别：记录到未知类型池
        self._unknown_types[raw_type] = self._unknown_types.get(raw_type, 0) + 1

        # 检查是否达到阈值，自动升级
        if self._unknown_types[raw_type] >= self._growth_threshold:
            self._promote_type(raw_type)

        return "Entity"

    def _promote_type(self, raw_type: str):
        """将未知类型升级为 learned 类型。"""
        # 生成 Kuzu Label：取中文类型的拼音或直接用原始类型
        kuzu_label = raw_type.replace(" ", "_")
        self._learned_types[raw_type] = kuzu_label
        logger.info(
            f"  📈 Schema 自动成长: 类型 '{raw_type}' 出现 "
            f"{self._unknown_types[raw_type]} 次，已升级为 learned 类型 → '{kuzu_label}'"
        )

    def get_all_types(self) -> dict[str, str]:
        """返回所有已知类型（builtin + learned）。"""
        all_types = dict(self._builtin)
        all_types.update(self._learned_types)
        return all_types

    def get_unknown_type_report(self) -> dict[str, int]:
        """返回未知类型池的统计报告。"""
        return dict(self._unknown_types)

    # ---- Relation Type ----

    def normalize_predicate(self, raw_predicate: str) -> str:
        """将中文谓词规范化为标准谓词。

        如果映射表中没有，返回原始谓词（不做修改）。
        """
        if not raw_predicate:
            return raw_predicate
        return self._relation_map.get(raw_predicate, raw_predicate)

    def is_known_predicate(self, predicate: str) -> bool:
        """判断谓词是否在规范化映射表中。"""
        return predicate in self._relation_map

    # ---- Build Fingerprint ----

    def compute_fingerprint(
        self,
        extract_prompt: str,
        validate_prompt: str,
        extract_model: str,
        validate_model: str,
        code_version: str = "1.0",
    ) -> str:
        """计算构建指纹，用于检测缓存是否失效。

        当内置 Schema、Prompt、模型、代码版本任一变化时，指纹也会变化。

        注意：learned_types 不参与指纹计算 —— 它在构建过程中自动成长并保存，
        若计入指纹会导致"下次启动指纹变化 → 缓存整体失效 → 全书重抽"，
        断点续传机制被 Schema 成长机制击穿。
        """
        components = [
            json.dumps(sorted(self._builtin.items()), sort_keys=True),
            json.dumps(sorted(self._relation_map.items()), sort_keys=True),
            extract_prompt,
            validate_prompt,
            extract_model,
            validate_model,
            code_version,
        ]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        """序列化为字典，用于持久化到缓存。"""
        return {
            "builtin_types": dict(self._builtin),
            "learned_types": dict(self._learned_types),
            "relation_map": dict(self._relation_map),
            "unknown_types": dict(self._unknown_types),
            "growth_threshold": self._growth_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Schema":
        """从字典恢复 Schema。"""
        schema = cls(
            builtin_types=data.get("builtin_types"),
            relation_map=data.get("relation_map"),
            growth_threshold=data.get("growth_threshold", 5),
        )
        schema._learned_types = data.get("learned_types", {})
        schema._unknown_types = data.get("unknown_types", {})
        return schema

    @classmethod
    def load_or_create(cls, cache_path: str, growth_threshold: int = 5) -> "Schema":
        """从缓存加载 Schema，如果不存在则创建新的。growth_threshold 以传入值为准。"""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"  从缓存加载 Schema: {len(data.get('learned_types', {}))} 个 learned 类型")
                schema = cls.from_dict(data)
                schema._growth_threshold = growth_threshold
                return schema
            except Exception as e:
                logger.warning(f"  Schema 缓存加载失败: {e}，将创建新的 Schema")
        return cls(growth_threshold=growth_threshold)

    def save(self, cache_path: str):
        """保存 Schema 到缓存。"""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"  Schema 缓存保存失败: {e}")