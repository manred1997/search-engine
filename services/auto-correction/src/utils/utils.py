import os
import json
import logging

import random
import numpy as np
import torch

from src.end_to_end.model.roberta_base import E2ESpellCheckRoberta
from transformers import (
    AutoTokenizer,
    RobertaConfig
)

MODEL_CLASSES = {
    "scRoberta": (RobertaConfig, E2ESpellCheckRoberta, AutoTokenizer)
}

MODEL_PATH_MAP = {
    "scRoberta": "vinai/phobert-base",
}


logger = logging.getLogger(__name__)


vowel = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
        ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
        ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
        ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
        ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
        ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
        ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
        ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
        ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
        ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
        ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
        ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]

vowel_to_idx = {}
for i in range(len(vowel)):
    for j in range(len(vowel[i]) - 1):
        vowel_to_idx[vowel[i][j]] = (i, j)

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_tokenizer(args):
    logger.info("Loadding Tokenizer")
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)

def _read_json_file(filename):
    logger.info("Reading json file from {}".format(filename))
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info("Reading done")
    return data

def _write_json_file(filename, data):
    logger.info("Writing json file at {}".format(filename))
    foldername = filename.split('/')[:-1]
    if foldername:
        foldername = "/".join(foldername)
        if not os.path.exists(foldername):
            os.mkdir(foldername)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("Writing done")
    
def _read_text_file(filename):
    logger.info("Reading text file from {}".format(filename))
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip())
    logger.info("Reading done")
    return data

def _write_text_file(filename, data):
    logger.info("Writing text file at {}".format(filename))
    assert type(data) is list
    foldername = filename.split('/')[:-1]
    if foldername:
        foldername = "/".join(foldername)
        if not os.path.exists(foldername):
            os.mkdir(foldername)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write("%s\n" % item)
    logger.info("Writing done")

def is_valid_vietnam_word(word):
    chars = list(word)
    vowel_index = -1
    for index, char in enumerate(chars):
        x, _ = vowel_to_idx.get(char, (-1, -1))
        if x != -1:
            if vowel_index == -1:
                vowel_index = index
            else:
                if index - vowel_index != 1:
                    return False
                vowel_index = index
    return True


def chunks(l, n):
    for i in range(0, len(l) - n + 1):
        yield l[i:i+n]