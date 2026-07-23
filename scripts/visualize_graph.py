"""知识图谱可视化：实体圆圈 + 关系连线，节点大小按度数缩放。

数据源：激活语料的 corpora/<slug>/data/graph_cache/graph_db.db
（SQLite 图构建缓存，与 Kuzu 图同源；语料由 RAG_CORPUS 选择）。
输出：仓库根目录 graph_visualization.html（pyvis 交互式页面）。

用法：
  python scripts/visualize_graph.py           # 默认展示度数 top-80 实体
  python scripts/visualize_graph.py 150       # 展示 top-150
"""
import os
import sqlite3
import sys
from collections import Counter

import networkx as nx
from pyvis.network import Network

# 实体类型（Kuzu Label）→ 颜色
TYPE_COLORS = {
    "Person": "#e74c3c",
    "Organization": "#3498db",
    "Location": "#2ecc71",
    "Event": "#e91e63",
    "Item": "#f39c12",
    "Concept": "#9b59b6",
    "CreativeWork": "#1abc9c",
    "Entity": "#95a5a6",
}


def load_graph_from_cache(db_path: str) -> nx.MultiDiGraph:
    """从 SQLite 图构建缓存读取实体类型与关系，构建 networkx 图。"""
    conn = sqlite3.connect(db_path)
    try:
        entity_rows = conn.execute(
            """SELECT canonical_name, type FROM entities
               WHERE canonical_name IS NOT NULL AND canonical_name != ''"""
        ).fetchall()
        relation_rows = conn.execute(
            "SELECT subject, predicate, object FROM relations"
        ).fetchall()
    finally:
        conn.close()

    # 实体类型映射：优先保留具体类型（非 Entity 兜底类型）
    entity_types: dict[str, str] = {}
    for name, etype in entity_rows:
        etype = etype or "Entity"
        if name not in entity_types or (entity_types[name] == "Entity" and etype != "Entity"):
            entity_types[name] = etype

    G = nx.MultiDiGraph()
    for subj, pred, obj in relation_rows:
        G.add_node(subj, type=entity_types.get(subj, "Entity"))
        G.add_node(obj, type=entity_types.get(obj, "Entity"))
        G.add_edge(subj, obj, label=pred)

    return G


def get_node_size(degree: int, min_degree: int, max_degree: int) -> int:
    if max_degree == min_degree:
        return 25
    return int(10 + (degree - min_degree) / (max_degree - min_degree) * 50)


def visualize(G: nx.MultiDiGraph, output_path: str, top_n: int = 80):
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    min_deg = min(degrees.values()) if degrees else 1

    # 取度数最高的 top_n 个节点
    top_nodes = set()
    if top_n > 0 and len(G.nodes) > top_n:
        top_nodes = {n for n, _ in Counter(degrees).most_common(top_n)}

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
        bgcolor="#ffffff",
        font_color="#333333",
    )

    # 使用 forceAtlas2Based 布局，比 barnesHut 更整洁
    net.set_options("""
    var options = {
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -120,
          "centralGravity": 0.005,
          "springLength": 450,
          "springConstant": 0.03,
          "damping": 0.4,
          "avoidOverlap": 0.8
        },
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "updateInterval": 25
        }
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.4 } },
        "smooth": { "type": "continuous" },
        "color": { "color": "#cccccc", "highlight": "#2B7CE9" },
        "width": 0.5,
        "selectionWidth": 1.5
      },
      "nodes": {
        "font": { "size": 12, "face": "Microsoft YaHei", "color": "#333333" },
        "borderWidth": 1,
        "borderWidthSelected": 3,
        "shapeProperties": {
          "borderRadius": 50
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true
      }
    }
    """)

    # 先添加所有节点
    visible_nodes: set[str] = set()
    for node in G.nodes():
        if top_n and top_nodes and node not in top_nodes:
            continue
        visible_nodes.add(node)
        deg = degrees.get(node, 1)
        size = get_node_size(deg, min_deg, max_deg)
        ntype = G.nodes[node].get("type", "Entity")
        color = TYPE_COLORS.get(ntype, "#95a5a6")
        net.add_node(
            node,
            label=node,
            title=f"{node}\n类型: {ntype}\n连接数: {deg}",
            size=size,
            color=color,
            borderWidth=1,
            borderWidthSelected=3,
        )

    # 边 —— 只添加两端都在可见集合中的边
    edges_added: set[tuple] = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in visible_nodes or v not in visible_nodes:
            continue
        edge_key = (u, v, data.get("label", ""))
        if edge_key in edges_added:
            continue
        edges_added.add(edge_key)
        net.add_edge(
            u, v,
            label=data.get("label", ""),
            title=f"{u} → {data.get('label', '')} → {v}",
            width=0.5,
            arrowStrikethrough=False,
        )

    # 自行生成 HTML 并以 UTF-8 写盘（避免 pyvis 在 Windows 上按 GBK 写文件）
    html = net.generate_html()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"可视化完成: {output_path}")
    print(f"  节点: {len(visible_nodes)}")
    print(f"  边: {len(edges_added)}")
    print(f"  最大度数: {max_deg}")

    # 打印度数 Top 10
    print("\n度数 Top 10 实体:")
    for name, cnt in Counter(degrees).most_common(10):
        print(f"  {name}: {cnt}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from rag import config

    db_path = os.path.join(
        os.path.dirname(config.GRAPH_DB_DIR), "graph_cache", "graph_db.db"
    )
    if not os.path.exists(db_path):
        print(f"图构建缓存不存在: {db_path}")
        print("请先运行 python scripts/build_full_graph.py 构建知识图谱")
        sys.exit(1)

    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 80

    print(f"加载图构建缓存: {db_path}")
    G = load_graph_from_cache(db_path)
    print(f"  实体: {len(G.nodes)}, 关系: {len(G.edges)}")

    output_path = os.path.join(os.path.dirname(__file__), "..", "graph_visualization.html")
    visualize(G, output_path, top_n=top_n)
