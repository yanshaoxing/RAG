"""
GraphRAG 数据模型 —— 使用 dataclass 替代 dict，提供类型安全和 IDE 自动补全。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    name: str
    type: str = "Entity"
    description: str = ""
    confidence: float = 0.0
    chunk_id: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.name.strip()) >= 2


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    subject_type: str = "Entity"
    object_type: str = "Entity"
    description: str = ""
    chunk_id: int = 0
    source_text: str = ""
    confidence: float = 0.0
    validated: bool = False
    extract_model: str = ""
    validate_model: str = ""

    @property
    def triple_key(self) -> tuple:
        return (self.subject, self.predicate, self.object)

    @property
    def is_valid(self) -> bool:
        return (
            len(self.subject.strip()) >= 2
            and len(self.object.strip()) >= 2
            and len(self.predicate.strip()) >= 2
        )


@dataclass
class ChunkResult:
    chunk_id: int
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    raw_text: str = ""

    @property
    def is_empty(self) -> bool:
        return len(self.relations) == 0


@dataclass
class SchemaTypeInfo:
    name: str
    category: str = "builtin"  # "builtin" | "learned"
    count: int = 0


@dataclass
class BuildMetrics:
    total_chunks: int = 0
    success_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    filtered_by_rules: int = 0
    filtered_by_llm: int = 0
    corrected_by_llm: int = 0
    merged_descriptions: int = 0
    canonicalized_entities: int = 0

    @property
    def avg_relations_per_chunk(self) -> float:
        if self.success_chunks == 0:
            return 0.0
        return self.total_relations / self.success_chunks

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  GraphRAG 构建统计",
            "=" * 50,
            f"  Chunk 总数:      {self.total_chunks}",
            f"  抽取成功:        {self.success_chunks}",
            f"  抽取失败:        {self.failed_chunks}",
            f"  实体总数:        {self.total_entities}",
            f"  关系总数:        {self.total_relations}",
            f"  规则过滤:        {self.filtered_by_rules}",
            f"  LLM 过滤:        {self.filtered_by_llm}",
            f"  LLM 修正:        {self.corrected_by_llm}",
            f"  Description 合并: {self.merged_descriptions}",
            f"  实体规范化:      {self.canonicalized_entities}",
            f"  平均每 Chunk:    {self.avg_relations_per_chunk:.1f} 条关系",
        ]
        return "\n".join(lines)