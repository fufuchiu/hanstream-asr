# 架构与边界

```text
PCM16 WAV -> mono waveform -> log-mel -> GRU -> frame log probabilities
                                              |              |
                                         CTC objective   greedy / beam
                                                             |
                                                     text / token spans
                                                             |
                                                     WER / CER analysis
```

CTC blank 默认为 0。路径先合并连续重复，再移除 blank；两个相同输出字符之间
必须存在 blank。强制对齐使用半开帧区间，无法对齐时明确失败。

特征采用 HTK mel 频率和自然对数，最后一帧补零。当前 WAV 读取只接受 PCM16，
多通道取平均；不静默重采样。能量端点检测是可解释基线，嘈杂环境应替换或校准。

中文推荐 CER；字符单位为 Unicode code point，英语 WER 按空白分词。
空参考文本的插入错误以 1 为分母，语料汇总采用总错误数除以总参考长度。
