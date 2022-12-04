import os
import json
import logging

import random
import numpy as np
import torch

import unicodedata

logger = logging.getLogger(__name__)

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


def normalize_string(string):
    string = unicodedata.normalize('NFKC', string).encode('ascii', 'ignore').decode('ascii')
    string = string.lower()
    string = ' '.join(string.split())
    return string