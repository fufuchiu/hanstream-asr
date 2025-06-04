from hanstream.metrics import score

for reference, hypothesis in [
    ('广州大学语音识别', '广州大学语言识别'),
    ('speech recognition', 'speech prediction'),
]:
    result = score(reference, hypothesis, 'char')
    print(reference, hypothesis, result, result.rate)
