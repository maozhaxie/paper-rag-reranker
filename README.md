# 论文 RAG 问答与中文重排微调

项目包含两个部分。论文 RAG 原型支持带页码引用的问答。独立的中文重排实验用于比较微调模型与 BM25。

## 论文问答原型

```text
PDF → 按页提取与切块 → 百炼向量模型 → Milvus 召回 → 重排 → 带页码的答案
```

在本地论文验证中，19 页文档切成 55 个文本块。向量归一化与重排调整后，一条时间范围问题的相关证据从原始排名 11 提升到 1。这是单个诊断案例。原版项目完成了 25 项离线检查，覆盖解析、检索逻辑和接口等环节。

### 本地运行

1. 在仓库根目录执行 `python -m pip install -r requirements-tested.txt`。
2. 复制 `.env.example` 为 `.env`。将有权使用的 PDF 放入 `papers/paper.pdf`，或设置 `PAPER_PATH`。填入自己的百炼 API Key 以启用索引和问答。
3. 使用现有 Milvus，或执行 `docker compose -f infra/docker-compose.yml up -d` 启动本地服务。
4. 执行 `python server.py`，打开终端显示的本地网址。先预览切块，再进行索引和问答。

使用 `python -m unittest discover -s tests` 运行仓库附带的离线重排检查。论文相关的评测题需针对所选 PDF 自行编写。

## 中文重排微调

在公开 T2Reranking 数据上对小型中文 RoBERTa 进行相关性二分类微调。训练使用 7,188 对样本，验证集用于选取检查点，在留出的 400 个问题上比较方法。

| 方法 | 测试 MRR@10 |
|---|---:|
| 同一编码器与随机分类头 | 0.4992 |
| 微调后的中文重排模型 | 0.5783 |
| 字符二元 BM25 | 0.6751 |

微调结果是三个种子的均值，仍低于 BM25。论文问答原型保留原有云端重排链路。

在独立 Python 3.10 环境安装适用于本机的 PyTorch 2.6，再安装 `rerank_training/requirements.txt`。从仓库根目录依次运行：

```sh
python rerank_training/prepare_resources.py
python rerank_training/train.py
python rerank_training/finalize_evidence.py
```

- [训练方案](rerank_training/protocol.md)
- [历史评估汇总](rerank_training/reference_metrics.json)
- [资源版本与来源](rerank_training/resources.json)

论文与 API 凭据由使用者在本地配置。训练资源从原发布方下载。
