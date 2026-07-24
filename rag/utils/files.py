"""
文件工具 —— JSON 原子写入与索引阶段完成标记。

每个索引阶段完成时在产物目录写入 _DONE.json 标记（含时间戳与统计信息）。
完成检测以标记文件为准 —— 仅凭目录存在无法区分"完成"与"中途中断"
（各阶段都是先建目录后写文件，中断会留下不完整目录导致静默加载损坏数据）。
"""

import json
import os
from datetime import datetime

DONE_MARKER = "_DONE.json"


def atomic_write_json(path: str, obj) -> None:
    """JSON 原子落盘：先写临时文件再 os.replace，避免中断留下半截文件。"""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def mark_stage_done(stage_dir: str, **info) -> None:
    """写入阶段完成标记。"""
    payload = {"completed_at": datetime.now().isoformat(), **info}
    atomic_write_json(os.path.join(stage_dir, DONE_MARKER), payload)


def stage_complete(stage_dir: str) -> bool:
    """检查阶段是否已完成（目录存在且含完成标记）。"""
    return os.path.exists(os.path.join(stage_dir, DONE_MARKER))


def read_stage_info(stage_dir: str) -> dict:
    """读取阶段完成标记内容（不存在或损坏时返回空字典）。"""
    try:
        with open(os.path.join(stage_dir, DONE_MARKER), "r", encoding="utf-8") as f:
            info = json.load(f)
        return info if isinstance(info, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
