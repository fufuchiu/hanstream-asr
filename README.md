# hanstream-asr

可复现的语音识别实验工具：声学特征、CTC 解码、中文英文评测与可训练基线。

Chen Yuxuan。

> 历史说明：本仓库由经过验证的补丁序列重建；2025 年至 2026 年 8 月的 Git 时间戳用于展示项目演进，不代表代码实际开发日期。

[![CI](https://github.com/fufuchiu/hanstream-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/fufuchiu/hanstream-asr/actions/workflows/ci.yml)

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

```bash
python -m hanstream score '广州大学语音识别' '广州大学语言识别' --unit char
python examples/stream_audio.py
```

语料使用 JSONL，每行包含 `id`、`audio`、`text`、`speaker`、`duration`，
可选 `sample_rate`，默认 16000。相对音频路径以清单文件所在目录为基准。

```bash
python -m hanstream audit corpus.jsonl --check-audio
```

## 模型实验

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python examples/train_tiny.py
```

提供可以训练的紧凑研究原型和完整数据链路。示例使用合成输入，验证损失下降、
梯度和掩码正确性；不包含经过大规模训练的权重，也没有真实语料成绩声明。

## 文档

- [架构与边界](docs/architecture.md)
- [模块索引](docs/api-reference.md)
- [复现实验与时间线说明](docs/reproducibility.md)
- [测试策略](docs/testing.md)
- [参与开发](CONTRIBUTING.md)

## 验证

```bash
python -m build
python -m pytest -q
ruff check .
ruff format --check .
```

MIT License · Chen Yuxuan
