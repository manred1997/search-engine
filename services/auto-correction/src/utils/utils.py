import os
import json
import logging

import random
import numpy as np
import torch

from src.end_to_end.model.roberta_base import E2ESubWordSpellCheckRoberta
from transformers import (
    AutoTokenizer,
    RobertaConfig
)

MODEL_CLASSES = {
    "scSubWordRoberta": (RobertaConfig, E2ESubWordSpellCheckRoberta, AutoTokenizer)
}

MODEL_PATH_MAP = {
    "scSubWordRoberta": "vinai/phobert-base",
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
    logger.info(f"Loadding Tokenizer from {args.model_name_or_path}")
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

def merge_subtokens(tokens):
    merged_tokens = []
    for token in tokens:
        if token.startswith("@@"):
            merged_tokens[-1] = merged_tokens[-1] + token[2:]
        else:
            merged_tokens.append(token)
    text = " ".join(merged_tokens)
    return text

def get_evals_base_on_ids(preds, targets, lengths=None):
    """
    given the predicted word idxs, this method computes the accuracy 
    by matching all values from 0 index to lengths index along each 
    batch example if have lengths param
    """

    correct = 0
    total = 0
    if lengths is not None:
        assert len(preds) == len(targets) == len(lengths)
        for pred, target, l in zip(preds, targets, lengths):
            correct += (pred[1:l+1] == target[1:l+1]).sum()
            total += l
    else:
        assert len(preds) == len(targets)
        for pred, target in zip(preds, targets):
            correct += (pred == target).sum()
            total += len(pred)
    return correct, total