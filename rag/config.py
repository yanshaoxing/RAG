"""
全局配置 —— 纯参数（端点 / 密钥 / 调参 / 开关），按模块分类，可选功能均有独立开关。

全部 LLM prompt 模板在 rag/prompts.py，本文件不放 prompt 文本。

内网端点与密钥支持环境变量覆盖（未设置时使用默认值）：
  RAG_EMBED_OLLAMA_BASE_URL / RAG_DAVY_BASE_URL / RAG_DAVY_API_KEY
  RAG_RERANK_BASE_URL / RAG_GRAPH_VALIDATE_LLM_BASE_URL / RAG_DEBUG
"""

import os

# ============================================================
# Embedding 模型
# ============================================================
EMBED_MODEL_NAME = "qwen3-embedding:8b"
# 远程 Ollama 服务地址
EMBED_OLLAMA_BASE_URL = os.environ.get("RAG_EMBED_OLLAMA_BASE_URL", "http://10.245.100.186:12434")
EMBED_BATCH_SIZE = 512
EMBED_VECTOR_DIM = 4096            # qwen3-embedding:8b 输出维度（用于 FAISS）
EMBED_TIMEOUT = 300.0              # embedding 请求超时（秒），独立于回答 LLM 超时

# embedding 断点续传：分段计算并落盘，向量阶段中断后续跑只补缺失段。
# 缓存目录独立于 data/vector/（该目录在阶段重建时会被整体删除）
EMBED_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embed_cache")
EMBED_CHECKPOINT_SEGMENT_NODES = 256   # 每段节点数（续传粒度；也限制了单次请求体大小）

# ============================================================
# HNSW 索引参数（替代 IndexFlatIP，大幅加速检索）
# ============================================================
HNSW_M = 16                      # 每个节点的最大连接数（越大越精确，但构建/检索越慢，推荐 16~64）
HNSW_EF_CONSTRUCTION = 200       # 构建时的搜索宽度（越大越精确，但构建越慢，推荐 100~500）
HNSW_EF_SEARCH = 64              # 检索时的搜索宽度（运行时参数，越大越精确但越慢，推荐 32~128）

# ============================================================
# 最终回答 LLM
# ============================================================
# provider: "ollama"（本地）| "davy"（云端）
ANSWER_PROVIDER = "davy"
ANSWER_STREAM_ENABLED = True       # 最终回答流式输出（False 时 query() 返回普通 Response）

# --- Ollama ---
ANSWER_OLLAMA_MODEL = "qwen3.5:9b"
ANSWER_OLLAMA_TEMPERATURE = 0.2
ANSWER_OLLAMA_TIMEOUT = 300.0

# --- Davy 云端连接（answer / rewrite 共用） ---
DAVY_BASE_URL = os.environ.get("RAG_DAVY_BASE_URL", "https://davy.labs.lenovo.com:5000/v1")
DAVY_MODEL_NAME = "nemotron-3-ultra"        # 当前使用的主模型
DAVY_CERT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "sha2rootca-ca_subca1.cer")
DAVY_API_KEY = os.environ.get("RAG_DAVY_API_KEY", "ZmU0Nt7kbSGH8SEIwvFCSVYwGDiTSQNMezbfLZW_3Aw")
DAVY_TIMEOUT = 120.0
DAVY_TEMPERATURE = 0.2
DAVY_MAX_RETRIES = 3               # 429/5xx/网络错误的最大重试次数（指数退避，尊重 Retry-After）
DAVY_RETRY_BASE_DELAY = 2.0        # 重试基础等待秒数（第 n 次重试等待 base * 2^n）

# ============================================================
# 查询重写（三路策略：NL改写 + HyDE + BM25关键词扩展）
# ============================================================
REWRITE_ENABLED = True           # 开关
# provider: "ollama" | "davy"
REWRITE_PROVIDER = "davy"
REWRITE_TEMPERATURE = 0.1        # 三路改写 LLM 温度（davy / ollama 通用）

# --- Ollama ---
REWRITE_OLLAMA_MODEL = "qwen3.5:9b"
REWRITE_OLLAMA_TIMEOUT = 180.0

# ============================================================
# 查询期并行（三路改写 / 子查询 / 图检索与主检索并发）
# ============================================================
# Davy 端点经压测 >2 并发会触发 429（DavyLLM 已有重试退避兜底），
# 故查询期 LLM 并发默认限 2。子查询与三路改写嵌套时最坏并发 = 两者乘积。
QUERY_REWRITE_MAX_CONCURRENCY = 2   # 三路改写（NL/HyDE/关键词）LLM 调用并发数，1=串行
SUBQUERY_MAX_CONCURRENCY = 2        # 分解后子查询的并行检索数，1=串行

# ============================================================
# 检索
# ============================================================
RETRIEVAL_TOP_K = 30             # 每路检索器原始召回量（向量 / BM25 共用）
RERANK_CANDIDATE_POOL_SIZE = 20  # RRF 融合后送入 reranker 的候选数
FINAL_TOP_K = 10                 # 最终返回量（rerank 后 or 无 reranker 时的最终 top-k）

# ============================================================
# RRF 融合
# ============================================================
RRF_K = 60.0                     # RRF 平滑参数

# ============================================================
# 摘要冗余过滤
# ============================================================
# 若摘要节点覆盖的原始 chunk 被召回比例 ≥ 此值，说明已被原文覆盖，删除该摘要节点
SUMMARY_REDUNDANCY_THRESHOLD = 0.5

# ============================================================
# 多路检索优化 —— 相邻分数比过滤（gap detection）
# ============================================================
GAP_THRESHOLD = 0.2              # 相邻块分数下降比例阈值
MIN_CANDIDATES = 10              # 每路过滤后最小候选块数量下限（防止过早截断导致块过少）
MAX_CANDIDATES = 30              # 每路过滤后最大候选块数量上限

# 每路最小分数阈值（分数量纲不同，各自独立）
VECTOR_MIN_SCORE = 0.1            # 向量检索（cosine similarity，约 P25 位置）
BM25_MIN_SCORE = 0.3              # BM25 检索（现有索引实测有效分数约 1.6~2.6，
                                  # 此下限基本不起过滤作用；重建索引后需按实际分布重新标定）

# ============================================================
# 查询分解（将复杂查询拆分为子查询，分别检索后合并）
# ============================================================
DECOMPOSE_ENABLED = True          # 开关
DECOMPOSE_MAX_SUB_QUERIES = 5     # 最大子查询数量
DECOMPOSE_LLM_TEMPERATURE = 0.1   # 分解/分类 LLM 温度

# ============================================================
# 重排序（bge-reranker-v2-m3：通过 vLLM /v1/rerank 端点调用）
# ============================================================
RERANK_ENABLED = True            # 开关
RERANK_BASE_URL = os.environ.get("RAG_RERANK_BASE_URL", "http://10.245.100.186:12345")
RERANK_MODEL_NAME = "bge_reranker_v2_m3"
RERANK_TIMEOUT = 30.0
RERANK_MAX_RETRIES = 1           # 瞬时故障（网络/429/5xx）快速重试次数，重试仍失败才降级 RRF
RERANK_TEXT_MAX_LENGTH = 8192     # 送入 reranker 的文本最大长度（bge-reranker-v2-m3 最大 8192 tokens）

# ============================================================
# 分块
# ============================================================
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 102                 # 单侧 overlap 上限（字符数），chunk_size 的 10%，相邻 chunk 重叠约此值

# ============================================================
# 摘要树
# ============================================================
SUMMARY_TREE_ENABLED = True       # 开关：启用层次化摘要树
SUMMARY_LLM_PROVIDER = "davy"     # 生成摘要使用的 LLM："ollama" | "davy"
SUMMARY_OLLAMA_MODEL = "qwen3.5:9b"   # provider=ollama 时的摘要模型
SUMMARY_OLLAMA_TIMEOUT = 300.0        # provider=ollama 时的摘要请求超时
SUMMARY_PARENT_RATIO = 0.20       # L2 小节摘要字数 = 输入文本总字数 × 此比例
SUMMARY_CHAPTER_RATIO = 0.15      # L3 章节摘要字数 = 输入文本总字数 × 此比例
SUMMARY_BOOK_CHARS = 500          # L4 全书摘要字数上限
SUMMARY_BATCH_SIZE = 20          # 每批次生成叶子摘要的数量（用于进度日志）
SUMMARY_MAX_CONCURRENCY = 2      # 摘要生成的最大并发请求数（经压测，>2 会触发 429）
SUMMARY_LEAF_BATCH_SIZE = 5      # 每次 LLM 调用合并的 chunk 数（减少 API 调用次数，约 2x 加速）
SUMMARY_LLM_TEMPERATURE = 0.1

# ============================================================
# 术语映射（通俗名 → 原文术语）
# ============================================================
TERM_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "terminology.json")

# ============================================================
# 持久化路径
# ============================================================
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector")
FAISS_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector", "faiss")
BM25_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "bm25")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SUMMARY_TREE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "summary_tree")
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chunks")
GRAPH_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "graph_db")

# ============================================================
# 知识图谱（PropertyGraph + Kuzu）
# ============================================================
GRAPH_ENABLED = True               # 开关：是否启用知识图谱构建与检索

# 图检索设置（图抽取按"节"进行，不再单独分块）
GRAPH_RETRIEVAL_MAX_TRIPLES = 20   # 每次图检索最多返回的三元组数量
GRAPH_RETRIEVAL_TOP_K = 5          # 匹配实体时取 top-k 个实体作为入口

# 实体/关系抽取 LLM 配置
GRAPH_EXTRACT_LLM_PROVIDER = "davy"   # "ollama" | "davy"
GRAPH_EXTRACT_MAX_CONCURRENCY = 2     # 抽取并发数
GRAPH_EXTRACT_BATCH_SIZE = 5          # 每批处理的 chunk 数

# 三元组校验 LLM 配置（不同模型交叉校验更可靠）
GRAPH_VALIDATE_LLM_PROVIDER = "davy"    # "ollama" | "davy"
GRAPH_VALIDATE_DAVY_MODEL = "gpt-oss-120b"  # 校验用 Davy 模型（与原主模型不同，交叉校验）
GRAPH_VALIDATE_LLM_MODEL = "qwen3.6:27b"  # 校验用 ollama 模型（仅 provider=ollama 时生效）
GRAPH_VALIDATE_LLM_BASE_URL = os.environ.get("RAG_GRAPH_VALIDATE_LLM_BASE_URL", "http://10.245.100.186:12434")
GRAPH_VALIDATE_LLM_TIMEOUT = 3600.0  # 1 小时，基本等于无限等待
GRAPH_VALIDATE_LLM_TEMPERATURE = 0.1

# 是否启用 LLM 二次校验
GRAPH_VALIDATE_ENABLED = True  # 开启后每个 chunk 做完规则过滤再送 LLM 校验

# 校验策略：置信度低于此阈值的关系才送 LLM 校验（抽取置信度范围 0.0~1.0，
# 由描述长度/类型识别度等启发式打分）。设为 >1.0（如 2.0）表示全部关系送校验。
GRAPH_VALIDATE_CONFIDENCE_THRESHOLD = 2.0

# Schema 自动成长阈值（未知类型出现次数达到此值后自动升级为 learned 类型）
GRAPH_SCHEMA_GROWTH_THRESHOLD = 5

# ============================================================
# 调试
# ============================================================
# 总调试开关：控制是否输出各步骤的 top-3/top-5 详情（环境变量 RAG_DEBUG=1 可开启）
DEBUG = os.environ.get("RAG_DEBUG", "0").lower() in ("1", "true", "yes")