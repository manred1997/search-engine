import json

input_file = './completions'

vocab = []
with open(input_file + '.dict', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        vocab.append(line.strip().replace('_', ' '))


with open(input_file + '.dict', 'w', encoding='utf-8') as f:
    for item in vocab:
        f.write('%s\n' %item)

with open(input_file + '.dict.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tmp = {}
for k, v in data.items():
    k = k.replace('_', ' ')
    tmp[k] = v

with open(input_file + '.dict.json', 'w', encoding='utf-8') as f:
    json.dump(tmp, f, ensure_ascii=False)
    