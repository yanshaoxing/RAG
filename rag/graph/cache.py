"""
SQLite 缓存模块 —— 替代 JSON 全量重写，支持断点续传、增量更新、快速恢复。

表结构：
  chunks:    每个 chunk 的处理状态
  entities:  抽取的实体（含 canonical name 和 description）
  relations: 抽取的关系（含置信度、校验状态）
  schema:    Schema 类型统计
  metadata:  构建元数据（fingerprint、版本等）
"""

import hashlib
import logging
import os
import sqlite3
import time
from typing import Optional

from .models import Entity, Relation

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    text_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    chunk_text TEXT,
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    canonical_name TEXT,
    type TEXT DEFAULT 'Entity',
    description TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    chunk_id INTEGER,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    subject_type TEXT DEFAULT 'Entity',
    object_type TEXT DEFAULT 'Entity',
    description TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    validated INTEGER DEFAULT 0,
    chunk_id INTEGER,
    source_text TEXT DEFAULT '',
    extract_model TEXT DEFAULT '',
    validate_model TEXT DEFAULT '',
    created_at REAL
);

CREATE TABLE IF NOT EXISTS schema_types (
    name TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT 'learned',
    count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object);
CREATE INDEX IF NOT EXISTS idx_relations_chunk ON relations(chunk_id);
CREATE INDEX IF NOT EXISTS idx_relations_triple ON relations(subject, predicate, object);
"""


class GraphCache:
    """SQLite 缓存，管理 GraphRAG 构建过程中的中间数据。"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ---- Chunk 管理 ----

    def chunk_exists(self, chunk_id: int, text: str) -> bool:
        """检查 chunk 是否已处理且文本未变。"""
        text_hash = self._hash_text(text)
        row = self._conn.execute(
            "SELECT text_hash, status FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return False
        return row[0] == text_hash and row[1] == "completed"

    def mark_chunk_processing(self, chunk_id: int, text: str):
        """标记 chunk 开始处理。"""
        text_hash = self._hash_text(text)
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO chunks (chunk_id, text_hash, status, chunk_text, created_at, updated_at)
               VALUES (?, ?, 'processing', ?, ?, ?)""",
            (chunk_id, text_hash, text[:500], now, now),
        )
        self._conn.commit()

    def mark_chunk_completed(self, chunk_id: int):
        """标记 chunk 处理完成。"""
        self._conn.execute(
            "UPDATE chunks SET status = 'completed', updated_at = ? WHERE chunk_id = ?",
            (time.time(), chunk_id),
        )
        self._conn.commit()

    def mark_chunk_failed(self, chunk_id: int):
        """标记 chunk 处理失败。"""
        self._conn.execute(
            "UPDATE chunks SET status = 'failed', updated_at = ? WHERE chunk_id = ?",
            (time.time(), chunk_id),
        )
        self._conn.commit()

    def get_completed_chunk_ids(self) -> set[int]:
        """获取所有已完成的 chunk ID。"""
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE status = 'completed'"
        ).fetchall()
        return {row[0] for row in rows}

    # ---- Entity 管理 ----

    def save_entities(self, entities: list[Entity]):
        """批量保存实体。"""
        now = time.time()
        rows = [
            (
                e.name, e.name, e.type, e.description,
                e.confidence, e.chunk_id, now,
            )
            for e in entities
        ]
        self._conn.executemany(
            """INSERT INTO entities (name, canonical_name, type, description, confidence, chunk_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def update_entity_description(self, name: str, description: str, confidence: float = 0.0):
        """更新实体的 description。"""
        self._conn.execute(
            "UPDATE entities SET description = ?, confidence = MAX(confidence, ?) WHERE canonical_name = ?",
            (description, confidence, name),
        )
        self._conn.commit()

    def update_entity_canonical(self, original_name: str, canonical_name: str):
        """将实体的所有记录指向 canonical name。"""
        self._conn.execute(
            "UPDATE entities SET canonical_name = ? WHERE name = ? OR canonical_name = ?",
            (canonical_name, original_name, original_name),
        )
        self._conn.execute(
            "UPDATE relations SET subject = ? WHERE subject = ?",
            (canonical_name, original_name),
        )
        self._conn.execute(
            "UPDATE relations SET object = ? WHERE object = ?",
            (canonical_name, original_name),
        )
        self._conn.commit()

    def get_all_entity_descriptions(self) -> dict[str, str]:
        """获取所有实体的 canonical name → description 映射。

        同一 canonical_name 可能有多行（多个 chunk 各存一行），取描述最长的一行 ——
        合并后的描述通过 update_entity_description 写回所有行，最长的即最完整；
        此前 GROUP BY 取非聚合列会返回任意一行，续跑时合并描述可能被原始描述遮蔽。
        """
        rows = self._conn.execute(
            """SELECT canonical_name, description, MAX(LENGTH(description))
               FROM entities WHERE description != '' AND canonical_name IS NOT NULL
               GROUP BY canonical_name"""
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    # ---- Relation 管理 ----

    def save_relations(self, relations: list[Relation]):
        """批量保存关系。"""
        now = time.time()
        rows = [
            (
                r.subject, r.predicate, r.object,
                r.subject_type, r.object_type,
                r.description, r.confidence,
                1 if r.validated else 0,
                r.chunk_id, r.source_text,
                r.extract_model, r.validate_model,
                now,
            )
            for r in relations
        ]
        self._conn.executemany(
            """INSERT INTO relations (subject, predicate, object, subject_type, object_type,
               description, confidence, validated, chunk_id, source_text,
               extract_model, validate_model, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def get_all_relations(self) -> list[Relation]:
        """获取所有关系。"""
        rows = self._conn.execute(
            """SELECT subject, predicate, object, subject_type, object_type,
                      description, confidence, validated, chunk_id, source_text,
                      extract_model, validate_model
               FROM relations"""
        ).fetchall()
        return [
            Relation(
                subject=row[0], predicate=row[1], object=row[2],
                subject_type=row[3], object_type=row[4],
                description=row[5], confidence=row[6],
                validated=bool(row[7]), chunk_id=row[8],
                source_text=row[9], extract_model=row[10],
                validate_model=row[11],
            )
            for row in rows
        ]

    def get_relations_by_chunk(self, chunk_id: int) -> list[Relation]:
        """获取指定 chunk 的所有关系。"""
        rows = self._conn.execute(
            """SELECT subject, predicate, object, subject_type, object_type,
                      description, confidence, validated, chunk_id, source_text,
                      extract_model, validate_model
               FROM relations WHERE chunk_id = ?""",
            (chunk_id,),
        ).fetchall()
        return [
            Relation(
                subject=row[0], predicate=row[1], object=row[2],
                subject_type=row[3], object_type=row[4],
                description=row[5], confidence=row[6],
                validated=bool(row[7]), chunk_id=row[8],
                source_text=row[9], extract_model=row[10],
                validate_model=row[11],
            )
            for row in rows
        ]

    def build_relation_map(self) -> dict[tuple, Relation]:
        """构建 O(1) 查找的关系映射。"""
        relations = self.get_all_relations()
        return {r.triple_key: r for r in relations}

    # ---- Metadata ----

    def set_metadata(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def is_fingerprint_valid(self, fingerprint: str) -> bool:
        """检查构建指纹是否匹配。"""
        stored = self.get_metadata("fingerprint")
        return stored == fingerprint

    def set_fingerprint(self, fingerprint: str):
        self.set_metadata("fingerprint", fingerprint)

    # ---- 工具 ----

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()