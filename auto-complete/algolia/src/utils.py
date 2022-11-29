import json

def read_json(path_file):
    with open(path_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data