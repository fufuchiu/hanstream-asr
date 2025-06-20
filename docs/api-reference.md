# 模块索引

| 模块 | 职责 |
|---|---|
| `audio` | PCM16 WAV、幅度变换、噪声与分帧 |
| `features` | HTK mel 特征、CMVN、delta 与掩码 |
| `text` | Unicode 规范化、字符词表与编解码 |
| `metrics` | 确定性编辑对齐、WER/CER 与混淆对 |
| `ctc` | 贪心、前缀束搜索与 Viterbi 强制对齐 |
| `manifest` | 语料校验、说话人隔离划分与时长批次 |
| `streaming` | 字节分帧、能量端点与稳定转写 |
| `model` | 可训练 GRU CTC 基线与严格损失校验 |

每个公共入口的参数签名、默认值和数据类字段都由 `tests/contracts/public-api.json` 固定。
输入校验在计算前完成；无效维度、非有限数字和越界 ID 抛出 `ValueError`。
