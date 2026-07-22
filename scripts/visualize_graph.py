"""知识图谱可视化：实体圆圈 + 关系连线，节点大小按度数缩放。"""
import json
import os
import sys
from collections import Counter

import networkx as nx
from pyvis.network import Network

# 实体类型 → 颜色
TYPE_COLORS = {
    "人物": "#e74c3c",
    "组织": "#3498db",
    "地点": "#2ecc71",
    "城市": "#27ae60",
    "国家": "#1abc9c",
    "概念": "#9b59b6",
    "物品": "#f39c12",
    "金钱数额": "#e67e22",
    "称号": "#1abc9c",
    "事件": "#e91e63",
    "未知": "#95a5a6",
}


def load_triples(cache_path: str) -> list[dict]:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    entity_types: dict[str, str] = {}

    for chunk in triples:
        for ent in chunk.get("entities", []):
            name = ent["name"]
            etype = ent.get("type", "未知")
            if name not in entity_types:
                entity_types[name] = etype
            elif entity_types[name] == "未知" and etype != "未知":
                entity_types[name] = etype

    for chunk in triples:
        for rel in chunk.get("relations", []):
            subj = rel["subject"]
            obj = rel["object"]
            pred = rel["predicate"]
            G.add_node(subj, type=entity_types.get(subj, "未知"))
            G.add_node(obj, type=entity_types.get(obj, "未知"))
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
        if top_n and node not in top_nodes:
            continue
        visible_nodes.add(node)
        deg = degrees.get(node, 1)
        size = get_node_size(deg, min_deg, max_deg)
        ntype = G.nodes[node].get("type", "未知")
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

    # pyvis 在 Windows 上默认用 GBK 写文件，monkey-patch 强制 UTF-8
    import builtins
    _orig_open = builtins.open
    def _utf8_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return _orig_open(file, mode, *args, **kwargs)
    builtins.open = _utf8_open
    try:
        net.save_graph(output_path)
    finally:
        builtins.open = _orig_open
    print(f"可视化完成: {output_path}")
    print(f"  节点: {len(visible_nodes)}")
    print(f"  边: {len(edges_added)}")
    print(f"  最大度数: {max_deg}")

    # 打印度数 Top 10
    print("\n度数 Top 10 实体:")
    for name, cnt in Counter(degrees).most_common(10):
        print(f"  {name}: {cnt}")


if __name__ == "__main__":
    cache_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "graph_triples_cache_graph_db.json"
    )
    if not os.path.exists(cache_path):
        print(f"缓存文件不存在: {cache_path}")
        sys.exit(1)

    print(f"加载缓存: {cache_path}")
    triples = load_triples(cache_path)
    print(f"  三元组 chunk 数: {len(triples)}")

    G = build_graph(triples)
    print(f"  实体: {len(G.nodes)}, 关系: {len(G.edges)}")

    output_path = os.path.join(os.path.dirname(__file__), "..", "graph_visualization.html")
    visualize(G, output_path, top_n=80)