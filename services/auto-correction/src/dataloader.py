from __future__ import absolute_import, division, print_function

import logging
import os
import re
import torch
from torch.utils.data import TensorDataset
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text=None, cap_label=None, punc_label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.cap_label = cap_label
        self.punc_label = punc_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, 
                    input_ids, 
                    attention_mask, 
                    words_lengths, 
                    attention_mask_label, 
                    cap_label, 
                    punc_label
                    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.words_lengths = words_lengths
        self.attention_mask_label = attention_mask_label
        self.cap_label = cap_label
        self.punc_label = punc_label

def readfile(filename,eos_marks=['PERIOD', 'QMARK']):
    paragraphs = []
    cap_labels = []
    punc_labels = []
    max_len_seq = 50
    with open(filename, mode="r", encoding="utf-8") as f:
        paragraph, cap_label , punc_label = [], [], []
        for line in f:
            line = line.lstrip().rstrip()
            word, cap, punc = line.split('\t')

            if len(paragraph) < max_len_seq:
                paragraph.append(word)
                cap_label.append(int(cap))
                punc_label.append(punc)
            if len(paragraph) == max_len_seq:

                i = len(paragraph) - 1
                while i > 0:

                    if punc_label[i] in eos_marks:
                        if i == (max_len_seq - 1):
                            paragraphs.append(paragraph)
                            cap_labels.append(cap_label)
                            punc_labels.append(punc_label)
                            paragraph, cap_label , punc_label = [], [], []
                        else:
                            paragraphs.append(paragraph[:i+1])
                            cap_labels.append(cap_label[:i+1])
                            punc_labels.append(punc_label[:i+1])
                            paragraph = paragraph[i + 1:]
                            cap_label = cap_label[i + 1:]
                            punc_label = punc_label[i + 1:]
                        break
                    else:
                        i = i - 1

                if len(paragraph) == max_len_seq:
                    paragraph, cap_label , punc_label = [], [], []
        
    return list(zip(paragraphs, cap_labels, punc_labels))


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class CapPuncProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return [ 'O', 'PERIOD', 'COMMA', 'QMARK','[CLS]', '[SEP]']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence,cap_label,punc_label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = ' '.join(sentence)
            cap_label = cap_label
            punc_label = punc_label
            examples.append(InputExample(guid=guid, text=text, cap_label=cap_label,punc_label=punc_label))
        return examples


def convert_examples_to_features(examples, punc_label_set, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    punc_label_map = {label: i for i, label in enumerate(punc_label_set)}

    features = []
    count = 0
    for (ex_index, example) in enumerate(examples):
        count+=1
        textlist = example.text.split(' ')
        cap_label_list = example.cap_label
        punc_label_list = example.punc_label

        if len(textlist) > max_seq_length:
            textlist = textlist[:max_seq_length-2]
            cap_label_list = cap_label_list[:max_seq_length-2]
            punc_label_list = punc_label_list[:max_seq_length-2]

        tokens = []
        cap_labels = []
        punc_labels = []
        words_lengths = []
        attention_mask_label = []
        attention_mask = []
        for i, word in enumerate(textlist):
            if re.search(r'([+-]?\d+[\.,]?)+', word) is not None or word.isnumeric():
                word = '<NUM>'
            
            token = tokenizer.tokenize(word)

            if not token:
                token = [tokenizer.unk_token]
            tokens.extend(token)
            words_lengths.append(len(token))
            cap_label = cap_label_list[i]
            punc_label = punc_label_list[i]
            cap_labels.append(cap_label)
            punc_labels.append(punc_label_map[punc_label])
            attention_mask_label.append(1)
        
        tokens += [tokenizer.sep_token]
        cap_labels += [0]
        punc_labels += [punc_label_map['[SEP]']]
        words_lengths += [1]
        attention_mask_label += [1]



        tokens = [tokenizer.cls_token] + tokens
        cap_labels = [0] + cap_labels
        punc_labels = [punc_label_map['[CLS]']] + punc_labels
        words_lengths = [1] + words_lengths
        attention_mask_label = [1] + attention_mask_label


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(cap_labels)
        if padding_length > 0:
            cap_labels = cap_labels + ([0] * padding_length)
            words_lengths = words_lengths + ([1]*padding_length)
            punc_labels = punc_labels + ([punc_label_map["O"]]*padding_length)
            attention_mask_label = attention_mask_label + ([0]*padding_length)


        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      words_lengths=words_lengths,
                                      attention_mask_label=attention_mask_label,
                                      cap_label=cap_labels,
                                      punc_label=punc_labels))
    return features

def pad_concat(sample, pad_value=0):
    final_tensor = []
    max_length = max([torch.tensor(i).size(0) for i in sample]) # max len in data
    for i in sample:
        i = torch.tensor(i,dtype=torch.long)
        pa = max_length - i.size(0)
        if pa > 0:
            tensor = torch.nn.functional.pad(i, (0,pa), "constant", pad_value)
            final_tensor.append(tensor)
        else:
            final_tensor.append(i)
    return torch.stack(final_tensor)

def load_and_cache_examples(args, punc_label_set, tokenizer, mode):
    processor = CapPuncProcessor()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_length
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif mode == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif mode == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(
            examples,punc_label_set, args.max_seq_length, tokenizer
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_input_ids = pad_concat(all_input_ids,tokenizer.pad_token_id)

    all_attention_mask = [f.attention_mask for f in features]
    all_attention_mask = pad_concat(all_attention_mask,0)
    
    
    all_words_lengths = torch.tensor([f.words_lengths for f in features], dtype=torch.long)
    
    all_attention_mask_label = torch.tensor([f.attention_mask_label for f in features], dtype=torch.long)
    all_cap_label = torch.tensor([f.cap_label for f in features], dtype=torch.long)
    all_punc_label = torch.tensor([f.punc_label for f in features], dtype=torch.long)


    dataset = TensorDataset(
        all_input_ids, all_words_lengths, all_attention_mask, all_attention_mask_label, all_cap_label, all_punc_label
    )
    return dataset
