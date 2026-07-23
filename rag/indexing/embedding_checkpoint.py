"""
Embedding 断点续传 —— 分段计算 embedding 并落盘，中断后续跑只补缺失段。

此前向量阶段直接 VectorStoreIndex(all_nodes)：5000 个节点崩在第 4999 个 = 全部重来
（原 embedding_progress.py 只打进度日志、不做续传）。现在：

  1. 节点按 EMBED_CHECKPOINT_SEGMENT_NODES 分段，每段调用一次
     embed_model.get_text_embedding_batch，结果存为 .npy（原子写入）；
  2. manifest.json 记录语料指纹（节点 id 序列 + 模型名的 md5）与已完成段；
     指纹不匹配（语料/模型变了）时整个缓存作废重建；
  3. 全部段就绪后把 embedding 填回节点（VectorStoreIndex 对已带 embedding
     的节点不再调用 embed 模型），向量索引持久化成功后由调用方清理缓存目录。

缓存目录独立于 data/vector/ —— 该目录在阶段重建时会被 staged_indexer 整体删除。
"""

import hashlib
import logging
import os
import shutil

import numpy as np
from llama_index.core.schema import MetadataMode

from rag import config
from rag.utils.files import atomic_write_json

logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"


def _fingerprint(nodes: list, model_name: str) -> str:
    """语料指纹：节点 id 序列 + 模型名。节点集合/顺序/模型任一变化都会使缓存作废。"""
    h = hashlib.md5()
    for node in nodes:
        h.update(node.node_id.encode("utf-8"))
        h.update(b"|")
    h.update(model_name.encode("utf-8"))
    return h.hexdigest()


def _load_manifest(path: str) -> dict:
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def _seg_path(cache_dir: str, seg_idx: int) -> str:
    return os.path.join(cache_dir, f"seg_{seg_idx:05d}.npy")


def embed_nodes_with_checkpoint(nodes: list, embed_model, label: str = "向量索引") -> None:
    """为节点批量计算 embedding（就地填充 node.embedding），分段落盘断点续传。"""
    if not nodes:
        return

    cache_dir = config.EMBED_CHECKPOINT_DIR
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, _MANIFEST)

    model_name = getattr(embed_model, "model_name", config.EMBED_MODEL_NAME)
    fp = _fingerprint(nodes, model_name)
    seg_size = config.EMBED_CHECKPOINT_SEGMENT_NODES
    num_segments = (len(nodes) + seg_size - 1) // seg_size

    manifest = _load_manifest(manifest_path)
    if manifest.get("fingerprint") != fp or manifest.get("segment_size") != seg_size:
        if manifest:
            logger.info("  embedding 缓存指纹不匹配（语料/模型/分段变化），作废重建")
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
        manifest = {"fingerprint": fp, "segment_size": seg_size, "segments": []}
        atomic_write_json(manifest_path, manifest)

    done_segments = set(manifest.get("segments", []))
    if done_segments:
        logger.info(f"  embedding 断点续传：{len(done_segments)}/{num_segments} 段已缓存")

    texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]

    for seg_idx in range(num_segments):
        start = seg_idx * seg_size
        end = min(start + seg_size, len(nodes))
        path = _seg_path(cache_dir, seg_idx)

        if seg_idx in done_segments and os.path.exists(path):
            embeddings = np.load(path)
        else:
            embeddings = np.asarray(
                embed_model.get_text_embedding_batch(texts[start:end], show_progress=False),
                dtype=np.float32,
            )
            if embeddings.shape[0] != end - start:
                raise RuntimeError(
                    f"embedding 数量不匹配: 段 {seg_idx} 期望 {end - start} 条，"
                    f"实际 {embeddings.shape[0]} 条"
                )
            # 原子写入：先写临时文件再 rename
            tmp_path = f"{path}.tmp.npy"
            np.save(tmp_path, embeddings)
            os.replace(tmp_path, path)
            manifest["segments"] = sorted(set(manifest.get("segments", [])) | {seg_idx})
            atomic_write_json(manifest_path, manifest)

            pct = int(end / len(nodes) * 100)
            msg = f"{label}: 已嵌入 {end}/{len(nodes)} ({pct}%)"
            print(f"    {msg}", flush=True)
            logger.info(msg)

        for i, node in enumerate(nodes[start:end]):
            node.embedding = embeddings[i].tolist()


def clear_checkpoint() -> None:
    """向量索引持久化成功后清理 embedding 缓存（约百 MB，无需长期保留）。"""
    shutil.rmtree(config.EMBED_CHECKPOINT_DIR, ignore_errors=True)
