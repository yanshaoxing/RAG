"""
全局配置 —— 纯参数（端点 / 密钥 / 调参 / 开关），按模块分类，可选功能均有独立开关。

全部 LLM prompt 模板在 rag/prompts.py，本文件不放 prompt 文本。

内网端点与密钥支持环境变量覆盖（未设置时使用默认值）：
  RAG_EMBED_OLLAMA_BASE_URL / RAG_DAVY_BASE_URL / RAG_DAVY_API_KEY
  RAG_RERANK_BASE_URL / RAG_GRAPH_VALIDATE_LLM_BASE_URL / RAG_DEBUG

全部 API key 只从环境变量 / 项目根目录 .env 读取（.env 已 gitignore，不入 git），
代码内不留明文兜底值：
  RAG_PUBLIC_CHAT_API_KEY / RAG_PUBLIC_EMBED_API_KEY / RAG_PUBLIC_RERANK_API_KEY
  RAG_DAVY_API_KEY

provider 选择与功能开关均可用环境变量覆盖，便于消融实验免改代码（见 _env_str/_env_bool）：
  RAG_EMBED_PROVIDER / RAG_ANSWER_PROVIDER / RAG_REWRITE_PROVIDER / RAG_RERANK_PROVIDER
  RAG_SUMMARY_LLM_PROVIDER / RAG_GRAPH_EXTRACT_LLM_PROVIDER / RAG_GRAPH_VALIDATE_LLM_PROVIDER
  RAG_REWRITE_ENABLED / RAG_DECOMPOSE_ENABLED / RAG_RERANK_ENABLED
  RAG_SUMMARY_TREE_ENABLED / RAG_GRAPH_ENABLED / RAG_GRAPH_VALIDATE_ENABLED
  RAG_ANSWER_STREAM_ENABLED / RAG_STRUCTURE_DETECT_ENABLED

多书语料：每本书一个目录 corpora/<slug>/（corpus.json + raw/ + terminology.json +
graph_rules.json + data/ 各阶段索引）。激活语料由 RAG_CORPUS 环境变量选择，
默认 DEFAULT_CORPUS。本文件的持久化路径全部从激活语料目录派生。
"""

import os


def _load_env_file() -> None:
    """加载项目根目录 .env 到环境变量（不覆盖已存在的变量）。

    公网 API key 不能硬编码入 git，统一放 .env；这里做最小解析，不引入 dotenv 依赖。
    """
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env_file()


def _env_str(name: str, default: str) -> str:
    """环境变量覆盖字符串配置（未设置或为空串时取默认值）。"""
    return os.environ.get(name, "").strip() or default


def _env_bool(name: str, default: bool) -> bool:
    """环境变量覆盖布尔开关（1/true/yes/on 为真，其余非空值为假，未设置取默认值）。

    消融实验（关摘要树 / 关图谱 / 关重排 …）靠这些开关免改代码切换。
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")

# ============================================================
# 多书语料（corpus）—— 每本书一个 corpora/<slug>/ 目录
# ============================================================
CORPORA_ROOT = os.path.join(os.path.dirname(__file__), "..", "corpora")
DEFAULT_CORPUS = "YaoYuanDeJiuShiZhu"
# 启动默认激活语料；运行期切换用 rag.corpus.set_active_corpus()
ACTIVE_CORPUS = os.environ.get("RAG_CORPUS", DEFAULT_CORPUS)

# 语料相关路径为【动态属性】（见文件末尾 __getattr__）：
# 随 rag.corpus 的激活语料实时计算，切书后 config.CHUNKS_DIR 等自动指向新语料。
# 名称 → 语料目录内的相对路径（"" = 语料根目录）
_CORPUS_RELATIVE_PATHS = {
    "CORPUS_DIR": "",
    "TERM_MAP_PATH": "terminology.json",
    "GRAPH_RULES_PATH": "graph_rules.json",
    "DATA_DIR": "raw",
    "PERSIST_DIR": "data/vector",
    "FAISS_PERSIST_DIR": "data/vector/faiss",
    "BM25_DIR": "data/bm25",
    "SUMMARY_TREE_DIR": "data/summary_tree",
    "CHUNKS_DIR": "data/chunks",
    "GRAPH_DB_DIR": "data/graph_db",
    "EMBED_CHECKPOINT_DIR": "data/embed_cache",
}


def __getattr__(name: str) -> str:
    """PEP 562：语料相关路径按当前激活语料实时计算（读取即最新，无缓存）。"""
    rel = _CORPUS_RELATIVE_PATHS.get(name)
    if rel is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from rag.corpus import get_active_slug  # 延迟导入避免循环依赖
    base = os.path.join(CORPORA_ROOT, get_active_slug())
    return os.path.join(base, *rel.split("/")) if rel else base

# ============================================================
# 公网（阿里云）端点 —— provider="aliyun" 时使用
# ============================================================
# 2026-07-23 起 chat / embedding / rerank 全部在同一个 cn-beijing workspace，
# 共用一把 key（三个 RAG_PUBLIC_*_API_KEY 变量保留，便于日后按角色拆分）。
# 该 workspace 专属部署无内容审核，小说正文实测正常（dashscope-us 公共端点
# 有内容审核会 400 拦截，勿改回）。当前模型与单价见项目根目录 LLM.txt。
ALIYUN_CHAT_BASE_URL = "https://ws-prbh7fipy7z0uzpu.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
ALIYUN_CHAT_API_KEY = os.environ.get("RAG_PUBLIC_CHAT_API_KEY", "")
ALIYUN_MAIN_MODEL = "qwen3.5-flash"        # 回答/改写/摘要/图谱抽取
ALIYUN_VALIDATE_MODEL = "qwen-flash"       # 图谱三元组校验（与抽取模型不同，交叉校验）
ALIYUN_CHAT_TIMEOUT = 120.0

# 思考（reasoning）开关 —— 对成本与延迟是数量级影响。
# 实测 qwen3.5-flash 同一条摘要请求：默认思考 = 5019 输出 token（其中 4987 是
# reasoning，占 99%）/ 45.7s；enable_thinking=False = 24 token / 0.7s，
# 概括质量无可见差异。构建期（摘要树数百次、图谱上千次调用）必须关闭，
# 否则输出计费与耗时都放大两个数量级。
# 注意：reasoning 不进 message.content，DavyLLM 的 <think> 剥离看不到它，但照样计费。
ALIYUN_ENABLE_THINKING = _env_bool("RAG_ALIYUN_ENABLE_THINKING", False)

# 上下文窗口与最大输出 —— 必须如实申报，否则 llama_index 按默认值 3900/256 处理。
# 后果：合成器（CompactAndRefine）按可用上下文 repack 参考资料，窗口报小会把本可
# 一次送入的上下文拆成多块 → 多轮 refine → 多次 LLM 调用（多花钱）、信息在
# 逐轮改写中损耗（答案变差）。全书查询送入约 1 万字正文，按 3900 必然触发多轮。
# 取值来自 LLM.txt：qwen3.5-flash / qwen-flash 上下文均为 1M。
# num_output 是"为输出预留的 token 数"（会从可用上下文中扣除），按 QA 回答的合理
# 上限取 8192 即可——模型实际最大输出 32K~64K，但我们的回答远短于此，预留过多只是
# 无谓压缩可用上下文。
ALIYUN_CONTEXT_WINDOW = int(_env_str("RAG_ALIYUN_CONTEXT_WINDOW", "1048576"))
ALIYUN_NUM_OUTPUT = int(_env_str("RAG_ALIYUN_NUM_OUTPUT", "8192"))

# ============================================================
# Embedding 模型
# ============================================================
# provider: "ollama"（内网远程 Ollama）| "aliyun"（阿里云 OpenAI 兼容端点）
EMBED_PROVIDER = _env_str("RAG_EMBED_PROVIDER", "aliyun")

# --- Ollama（内网） ---
EMBED_MODEL_NAME = "qwen3-embedding:8b"
# 远程 Ollama 服务地址
EMBED_OLLAMA_BASE_URL = os.environ.get("RAG_EMBED_OLLAMA_BASE_URL", "http://10.245.100.186:12434")
EMBED_BATCH_SIZE = 512

# --- 阿里云 ---
ALIYUN_EMBED_BASE_URL = "https://ws-prbh7fipy7z0uzpu.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
ALIYUN_EMBED_API_KEY = os.environ.get("RAG_PUBLIC_EMBED_API_KEY", "")
ALIYUN_EMBED_MODEL = "qwen3.7-text-embedding"   # 1024 维（cn-beijing workspace 上可用）
ALIYUN_EMBED_BATCH_SIZE = 10               # DashScope 兼容模式单次请求最多 10 条文本

# FAISS 向量维度随 embedding provider 切换（两模型维度不同，切换后必须重建语料的 data/vector/）
EMBED_VECTOR_DIM = 1024 if EMBED_PROVIDER == "aliyun" else 4096

# 当前生效的嵌入模型名 —— 写入向量阶段完成标记，加载时比对。
# ⚠️ 维度相同 ≠ 可复用：换模型后向量空间不同，旧索引与新查询向量的相似度无意义，
# 必须重建 data/vector/。此处的比对就是为了让这种情况显式报错而非静默给出垃圾结果。
ACTIVE_EMBED_MODEL_NAME = ALIYUN_EMBED_MODEL if EMBED_PROVIDER == "aliyun" else EMBED_MODEL_NAME
EMBED_TIMEOUT = 300.0              # embedding 请求超时（秒），独立于回答 LLM 超时

# embedding 断点续传：分段计算并落盘，向量阶段中断后续跑只补缺失段。
# 缓存目录 EMBED_CHECKPOINT_DIR（动态属性）独立于向量阶段目录
# （该目录在阶段重建时会被整体删除），随语料切换
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
# provider: "ollama"（本地）| "davy"（内网云端）| "aliyun"（公网阿里云）
ANSWER_PROVIDER = _env_str("RAG_ANSWER_PROVIDER", "aliyun")
# 最终回答流式输出（False 时 query() 返回普通 Response）
ANSWER_STREAM_ENABLED = _env_bool("RAG_ANSWER_STREAM_ENABLED", True)

# --- Ollama ---
ANSWER_OLLAMA_MODEL = "qwen3.5:9b"
ANSWER_OLLAMA_TEMPERATURE = 0.2
ANSWER_OLLAMA_TIMEOUT = 300.0

# --- Davy 云端连接（answer / rewrite 共用） ---
DAVY_BASE_URL = os.environ.get("RAG_DAVY_BASE_URL", "https://davy.labs.lenovo.com:5000/v1")
DAVY_MODEL_NAME = "nemotron-3-ultra"        # 当前使用的主模型
DAVY_CERT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "sha2rootca-ca_subca1.cer")
# 密钥不留明文兜底值：未设置时为空串，工厂函数会在选用 davy provider 时报错提示
DAVY_API_KEY = os.environ.get("RAG_DAVY_API_KEY", "")
DAVY_TIMEOUT = 120.0
DAVY_TEMPERATURE = 0.2
DAVY_MAX_RETRIES = 3               # 429/5xx/网络错误的最大重试次数（指数退避，尊重 Retry-After）
DAVY_RETRY_BASE_DELAY = 2.0        # 重试基础等待秒数（第 n 次重试等待 base * 2^n）

# ============================================================
# 查询重写（三路策略：NL改写 + HyDE + BM25关键词扩展）
# ============================================================
REWRITE_ENABLED = _env_bool("RAG_REWRITE_ENABLED", True)      # 开关
# provider: "ollama" | "davy" | "aliyun"
REWRITE_PROVIDER = _env_str("RAG_REWRITE_PROVIDER", "aliyun")
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
# 三路检索（向量NL / 向量HyDE / BM25）并发数：两路向量各含一次公网 embedding 往返、
# 彼此独立，串行是纯浪费；embedding 端点无 429 压测约束，默认 3 路全并发。
RETRIEVAL_ROUTE_MAX_CONCURRENCY = 3

# ============================================================
# 检索
# ============================================================
RETRIEVAL_TOP_K = 30             # 每路检索器原始召回量（向量 / BM25 共用）
RERANK_CANDIDATE_POOL_SIZE = 20  # RRF 融合后送入 reranker 的候选数
FINAL_TOP_K = 10                 # 最终返回量（rerank 后 or 无 reranker 时的最终 top-k）

# ---- 查询级缓存（P2-4）----
# 完全相同的问题重复提问会重跑整条检索管线（3 次改写 LLM + 2 次 embedding + rerank）。
# 开启后 HybridRetriever 按 (语料 slug + 原始 query + 检索 config 指纹) 缓存【检索结果】
# （不缓存回答——回答仍流式生成）。默认关，评测/演示脚本按需打开。
QUERY_CACHE_ENABLED = _env_bool("RAG_QUERY_CACHE_ENABLED", False)
QUERY_CACHE_MAX_SIZE = 128       # LRU 容量上限（按语料+查询+指纹计）

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
DECOMPOSE_ENABLED = _env_bool("RAG_DECOMPOSE_ENABLED", True)   # 开关
DECOMPOSE_MAX_SUB_QUERIES = 5     # 最大子查询数量
DECOMPOSE_LLM_TEMPERATURE = 0.1   # 分解/分类 LLM 温度

# ============================================================
# 重排序
# ============================================================
RERANK_ENABLED = _env_bool("RAG_RERANK_ENABLED", True)          # 开关
# provider: "vllm"（内网 vLLM /v1/rerank）| "aliyun"（阿里云 MaaS rerank 端点，需 Bearer 鉴权）
RERANK_PROVIDER = _env_str("RAG_RERANK_PROVIDER", "aliyun")

# --- vLLM（内网，bge-reranker-v2-m3） ---
RERANK_BASE_URL = os.environ.get("RAG_RERANK_BASE_URL", "http://10.245.100.186:12345")
RERANK_MODEL_NAME = "bge_reranker_v2_m3"

# --- 阿里云（qwen3-rerank；响应格式与 vLLM 相同：results[].index + relevance_score） ---
ALIYUN_RERANK_URL = "https://ws-prbh7fipy7z0uzpu.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks"
ALIYUN_RERANK_API_KEY = os.environ.get("RAG_PUBLIC_RERANK_API_KEY", "")
ALIYUN_RERANK_MODEL = "qwen3-rerank"
RERANK_TIMEOUT = 30.0
RERANK_MAX_RETRIES = 1           # 瞬时故障（网络/429/5xx）快速重试次数，重试仍失败才降级 RRF
RERANK_TEXT_MAX_LENGTH = 8192     # 送入 reranker 的文本最大长度（bge-reranker-v2-m3 最大 8192 tokens）

# ============================================================
# 分块
# ============================================================
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 102                 # 单侧 overlap 上限（字符数），chunk_size 的 10%，相邻 chunk 重叠约此值

# ============================================================
# 章节结构 LLM 检测（rag/ingestion/structure_detector.py）
# ============================================================
# 触发条件：语料档案无 chapter_pattern 字段 且 内置章节正则全文零命中（新书首次入库）。
# 检测结果经确定性校验后写回 corpora/<slug>/corpus.json，之后构建不再调 LLM。
STRUCTURE_DETECT_ENABLED = _env_bool("RAG_STRUCTURE_DETECT_ENABLED", True)
STRUCTURE_SAMPLE_HEAD_CHARS = 3000   # 采样：全文开头字符数
STRUCTURE_SAMPLE_SLICE_CHARS = 1000  # 采样：全文 25%/50%/75% 三处各取字符数
STRUCTURE_MIN_SECTIONS = 3           # 校验：正则切出的章节数下限（过少视为正则无效）
STRUCTURE_MAX_SECTIONS = 1000        # 校验：章节数上限（过多说明正则过宽、误匹配正文）
STRUCTURE_MAX_TITLE_CHARS = 50       # 校验：章节标题行长度上限（超长说明切在了正文中间）

# ============================================================
# 摘要树
# ============================================================
SUMMARY_TREE_ENABLED = _env_bool("RAG_SUMMARY_TREE_ENABLED", True)   # 开关：启用层次化摘要树
# 生成摘要使用的 LLM："ollama" | "davy" | "aliyun"
SUMMARY_LLM_PROVIDER = _env_str("RAG_SUMMARY_LLM_PROVIDER", "aliyun")
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
# 语料级资产与持久化路径 —— 均为动态属性（见文件开头 _CORPUS_RELATIVE_PATHS），
# 随激活语料实时派生：corpora/<slug>/{terminology.json, graph_rules.json, raw/, data/*}
# ============================================================

# ============================================================
# 知识图谱（PropertyGraph + Kuzu）
# ============================================================
GRAPH_ENABLED = _env_bool("RAG_GRAPH_ENABLED", True)   # 开关：是否启用知识图谱构建与检索

# 图检索设置（图抽取按"节"进行，不再单独分块）
GRAPH_RETRIEVAL_MAX_TRIPLES = 20   # 每次图检索最多返回的三元组数量
GRAPH_RETRIEVAL_TOP_K = 5          # 匹配实体时取 top-k 个实体作为入口

# 实体/关系抽取 LLM 配置
GRAPH_EXTRACT_LLM_PROVIDER = _env_str("RAG_GRAPH_EXTRACT_LLM_PROVIDER", "aliyun")  # "ollama" | "davy" | "aliyun"
# 抽取/校验 worker 并发数（记账在主线程串行）。主线程的 merge/canonicalize
# 也是 LLM 调用，实际 LLM 并发最坏 = 此值 + 1（Davy >2 并发 429，有重试兜底）
GRAPH_EXTRACT_MAX_CONCURRENCY = 2

# 三元组校验 LLM 配置（不同模型交叉校验更可靠）
# "ollama" | "davy" | "aliyun"（aliyun 用 ALIYUN_VALIDATE_MODEL）
GRAPH_VALIDATE_LLM_PROVIDER = _env_str("RAG_GRAPH_VALIDATE_LLM_PROVIDER", "aliyun")
GRAPH_VALIDATE_DAVY_MODEL = "gpt-oss-120b"  # 校验用 Davy 模型（与原主模型不同，交叉校验）
GRAPH_VALIDATE_LLM_MODEL = "qwen3.6:27b"  # 校验用 ollama 模型（仅 provider=ollama 时生效）
GRAPH_VALIDATE_LLM_BASE_URL = os.environ.get("RAG_GRAPH_VALIDATE_LLM_BASE_URL", "http://10.245.100.186:12434")
GRAPH_VALIDATE_LLM_TIMEOUT = 3600.0  # 1 小时，基本等于无限等待
GRAPH_VALIDATE_LLM_TEMPERATURE = 0.1

# 是否启用 LLM 二次校验
GRAPH_VALIDATE_ENABLED = _env_bool("RAG_GRAPH_VALIDATE_ENABLED", True)  # 开启后每个 chunk 做完规则过滤再送 LLM 校验

# 校验策略：置信度低于此阈值的关系才送 LLM 校验（抽取置信度范围 0.0~1.0，
# 由描述长度/类型识别度等启发式打分）。设为 >1.0（如 2.0）表示全部关系送校验。
GRAPH_VALIDATE_CONFIDENCE_THRESHOLD = 2.0

# Schema 自动成长阈值（未知类型出现次数达到此值后自动升级为 learned 类型）
GRAPH_SCHEMA_GROWTH_THRESHOLD = 5

# ============================================================
# 用量计量（rag/metering.py）
# ============================================================
# token 取自服务端响应的 usage 字段（计费同源），不做本地分词估算。
METERING_ENABLED = _env_bool("RAG_METERING_ENABLED", True)

# 模型单价（元/百万 token），来源：项目根目录 LLM.txt。
# cached_input = 显式缓存命中价（缺省时按 input 计）。
# 表中没有的模型不参与费用估算（汇总里显示「单价未知」，不静默算 0）。
MODEL_PRICES = {
    "qwen3.5-flash":          {"input": 0.2,  "output": 2.0, "cached_input": 0.02},
    "qwen-flash":             {"input": 0.15, "output": 1.5, "cached_input": 0.015},
    "qwen3.7-text-embedding": {"input": 0.5},
    "qwen3-rerank":           {"input": 0.5},
}

# ============================================================
# 调试
# ============================================================
# 总调试开关：控制是否输出各步骤的 top-3/top-5 详情（环境变量 RAG_DEBUG=1 可开启）
DEBUG = os.environ.get("RAG_DEBUG", "0").lower() in ("1", "true", "yes")