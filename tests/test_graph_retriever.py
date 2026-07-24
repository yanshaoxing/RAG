"""rag/graph/graph_retriever.py 单测 —— LLM 抽实体 → Kuzu 参数化查询 → 邻居遍历。

用假 LLM + 假图谱存储覆盖，不依赖真实 Kuzu / 模型。重点锁三件事：
  1. 可用性短路（图或 LLM 缺失、查询无具名实体、图无命中都返回空串）；
  2. 参数化查询（实体名走 param_map，绝不字符串插值进 Cypher，防注入/引号报错）；
  3. 健壮性（单个实体查询抛异常不拖垮整体、LLM 抽取失败静默返回空）。
"""

from rag import config
from rag.graph.graph_retriever import GraphRetriever


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeLLM:
    """complete() 返回预设的 JSON 文本；或在 raise_on 命中时抛异常。"""

    def __init__(self, text='["丁元英"]', raise_exc=None):
        self._text = text
        self._raise = raise_exc
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self._raise is not None:
            raise self._raise
        return _FakeResp(self._text)


class _FakeStore:
    """模拟 KuzuPropertyGraphStore.structured_query：按 param_map['entity'] 子串匹配。"""

    def __init__(self, triples=None, raise_for=None):
        # triples: list[(subject, predicate, object)]
        self._triples = triples or []
        self._raise_for = raise_for or set()
        self.calls = []  # 记录 (query_str, param_map)

    def structured_query(self, query_str, param_map=None):
        entity = (param_map or {}).get("entity")
        self.calls.append((query_str, dict(param_map or {})))
        if entity in self._raise_for:
            raise RuntimeError(f"Binder 异常：{entity}")
        rows = []
        for s, p, o in self._triples:
            if entity is not None and (entity in s or entity in o):
                rows.append({"subject": s, "predicate": p, "object": o})
        return rows


class _FakeIndex:
    def __init__(self, store):
        self.property_graph_store = store


def _retriever(llm=None, triples=None, raise_for=None):
    store = _FakeStore(triples=triples, raise_for=raise_for)
    index = _FakeIndex(store)
    r = GraphRetriever(graph_index=index, llm=llm or _FakeLLM())
    return r, store


# ---------- 可用性短路 ----------

class TestAvailability:
    def test_无图谱不可用(self):
        r = GraphRetriever(graph_index=None, llm=_FakeLLM())
        assert r.is_available is False
        assert r.retrieve("丁元英是谁") == ""

    def test_无llm不可用(self):
        r, _ = _retriever()
        r._llm = None
        assert r.is_available is False
        assert r.retrieve("丁元英是谁") == ""

    def test_查询无具名实体返回空(self):
        r, store = _retriever(llm=_FakeLLM(text="[]"))
        assert r.retrieve("这本书讲了什么") == ""
        assert store.calls == []   # 没实体就不该查图

    def test_图无命中返回空(self):
        r, _ = _retriever(llm=_FakeLLM(text='["查无此人"]'), triples=[("丁元英", "认识", "韩楚风")])
        assert r.retrieve("查无此人是谁") == ""


# ---------- 抽实体 ----------

class TestEntityExtraction:
    def test_过滤单字实体(self):
        # 单字实体（len < 2）会被丢弃，避免噪声匹配
        r = GraphRetriever(graph_index=None, llm=_FakeLLM(text='["丁", "丁元英"]'))
        assert r._extract_entities_from_query("q") == ["丁元英"]

    def test_去除首尾空白(self):
        r = GraphRetriever(graph_index=None, llm=_FakeLLM(text='["  丁元英  "]'))
        assert r._extract_entities_from_query("q") == ["丁元英"]

    def test_llm抛异常返回空列表(self):
        r = GraphRetriever(graph_index=None, llm=_FakeLLM(raise_exc=RuntimeError("超时")))
        assert r._extract_entities_from_query("q") == []

    def test_llm输出非json返回空(self):
        r = GraphRetriever(graph_index=None, llm=_FakeLLM(text="我无法回答"))
        assert r._extract_entities_from_query("q") == []


# ---------- 图搜索：参数化 / 去重 / 限额 / 健壮 ----------

class TestGraphSearch:
    def test_命中并格式化三元组(self):
        r, _ = _retriever(
            llm=_FakeLLM(text='["丁元英"]'),
            triples=[("丁元英", "好友", "韩楚风"), ("丁元英", "指点", "王庙村")],
        )
        out = r.retrieve("丁元英认识谁")
        assert "丁元英 → 好友 → 韩楚风" in out
        assert "丁元英 → 指点 → 王庙村" in out

    def test_实体名走参数不做字符串插值(self):
        # 防注入的核心保证：Cypher 里是占位符 $entity，实体名只出现在 param_map。
        # 用单引号构造注入串（保持 JSON 合法，避免被 json_repair 改写掩盖测试意图）。
        evil = "张三' OR 1=1 -- 删库"
        r, store = _retriever(llm=_FakeLLM(text=f'["{evil}"]'))
        r.retrieve("q")
        query_str, param_map = store.calls[0]
        assert "$entity" in query_str
        assert evil not in query_str          # 绝不能拼进查询串
        assert param_map["entity"] == evil    # 只经参数传递

    def test_三元组去重(self):
        # 同一三元组从多个实体入口重复命中，只保留一份
        r, _ = _retriever(
            llm=_FakeLLM(text='["丁元英", "韩楚风"]'),
            triples=[("丁元英", "好友", "韩楚风")],
        )
        out = r.retrieve("丁元英和韩楚风")
        assert out.count("丁元英 → 好友 → 韩楚风") == 1

    def test_单实体查询异常不拖垮整体(self, monkeypatch):
        # 一个实体触发 Binder 异常，另一个仍应正常返回
        r, store = _retriever(
            llm=_FakeLLM(text='["坏实体", "丁元英"]'),
            triples=[("丁元英", "好友", "韩楚风")],
            raise_for={"坏实体"},
        )
        out = r.retrieve("q")
        assert "丁元英 → 好友 → 韩楚风" in out
        assert len(store.calls) == 2   # 坏实体没有中断对第二个实体的查询

    def test_入口实体数受top_k限制(self, monkeypatch):
        monkeypatch.setattr(config, "GRAPH_RETRIEVAL_TOP_K", 2)
        r, store = _retriever(llm=_FakeLLM(text='["a1", "a2", "a3", "a4"]'))
        r.retrieve("q")
        assert len(store.calls) == 2   # 只查前 2 个实体

    def test_三元组总数受max_triples限制(self, monkeypatch):
        monkeypatch.setattr(config, "GRAPH_RETRIEVAL_MAX_TRIPLES", 3)
        triples = [("丁元英", "关系", f"对象{i}") for i in range(10)]
        r, _ = _retriever(llm=_FakeLLM(text='["丁元英"]'), triples=triples)
        out = r.retrieve("q")
        assert len(out.strip().splitlines()) == 3
