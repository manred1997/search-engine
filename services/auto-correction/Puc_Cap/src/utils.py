import logging
import os
import random

import numpy as np
import torch

from model import ViHnBERT  
from transformers import (
    AutoTokenizer,
    RobertaConfig
)

from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


MODEL_CLASSES = {
    "vihnbert": (RobertaConfig, ViHnBERT, AutoTokenizer)
}

MODEL_PATH_MAP = {
    "vihnbert": "/workspace/vinbrain/vutran/Backbone/HnBERTvn"
}


def parse_data(file_path, eos_marks, max_len_para=100):
    """
    :param file_path: text file path that contains word and capitalization/punctuations separated by tab in lines
    :param max_len_para: max sequence lenght of paragrapth
    :return: paragraphs, cap_labels, pun_labels
    """

    paragraphs = []

    cap_labels = []
    pun_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        paragraph, cap_label, pun_label = [], [], []
        for line in f.readlines():
            line = line.strip()
            word, cap, pun = line.split('\t')

            if len(paragraph) < max_len_para:
                paragraph.append(word)
                cap_label.append(cap)
                pun_label.append(pun)

            if len(paragraph) == max_len_para:
                i = len(paragraph) - 1
                while i > 0:
                    if pun_label[i] in eos_marks:
                        if i == (max_len_para - 1):
                            paragraphs.append(paragraph)
                            cap_labels.append(cap_label)
                            pun_labels.append(pun_label)
                            paragraph, cap_label , pun_label = [], [], []
                        else:
                            paragraphs.append(paragraph[:i+1])
                            cap_labels.append(cap_label[:i+1])
                            pun_labels.append(pun_label[:i+1])
                            paragraph = paragraph[i + 1:]
                            cap_label = cap_label[i + 1:]
                            pun_label = pun_label[i + 1:]
                        break
                    else:
                        i -= 1
                if len(paragraph) == max_len_para:
                    paragraph, cap_label , pun_label = [], [], []
    
    return list(zip(paragraphs, cap_labels, pun_labels))

def get_pun_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.data_dir, args.punc_label_file), "r", encoding="utf-8")
    ]

def get_cap_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.data_dir, args.cap_label_file), "r", encoding="utf-8")
    ]

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)



def compute_metrics(cap_preds, cap_labels, pun_preds, pun_labels, loss_coef):

    assert len(cap_preds) == len(cap_labels) == len(pun_preds) == len(pun_labels)
    print(f'[EVALUATE] Capitalization Prediction Task')
    print(classification_report(cap_preds, cap_preds, digits=4))

    print(f'[EVALUATE] Punctuation Prediction Task')
    print(classification_report(pun_preds, pun_labels, digits=4))

    cap_f1 = f1_score(cap_labels, cap_preds)
    pun_f1 = f1_score(pun_labels, pun_preds)

    results = {
        "cap_precision": precision_score(cap_labels, cap_preds),
        "cap_recall": recall_score(cap_labels, cap_preds),
        "cap_f1": f1_score(cap_labels, cap_preds),
        "pun_precision": precision_score(pun_labels, pun_preds),
        "pun_recall": recall_score(pun_labels, pun_preds),
        "pun_f1": f1_score(pun_labels, pun_preds),
        "mean_f1": mean_f1(cap_f1, pun_f1, loss_coef),
    }
    return results

def mean_f1(cap_f1, pun_labels, loss_coef=0.5):
    return ((10*loss_coef)*cap_f1+(10*(1-loss_coef))*pun_labels)/10