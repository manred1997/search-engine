

import os
import re
import logging

import torch
from torch.utils.data import TensorDataset

from utils import parse_data

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text=None, cap_label=None, punc_label=None):
        """Constructs a InputExample.
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
                token_type_ids,
                cap_label, 
                punc_label
                ):

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

        self.cap_label = cap_label
    
        self.punc_label = punc_label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return parse_data(input_file)


class CapPuncProcessor(DataProcessor):
    def __init__(self, args):
        self.args = args

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, cap_label, punc_label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = sentence
            cap_label = cap_label
            punc_label = punc_label
            examples.append(InputExample(guid=guid, text=text, cap_label=cap_label,punc_label=punc_label))
        return examples
    
    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            lines=self._read_file(
                os.path.join(data_path, 'data.txt')
            ),
            set_type=mode
        )



def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    pad_token_label_id=-100,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True
    ):

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        words = example.text
        cap_gt = example.cap_label
        punc_gt = example.punc_label

        tokens = []

        cap_labels = []
        punc_labels = []

        for word, cap, punc in zip(words, cap_gt, punc_gt):
            # if re.search(r'([+-]?\d+[\.,]?)+', word) is not None or word.isnumeric():
            #     word = '<NUM>'
            
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

            cap_labels.extend([int(cap)] + [int(cap)] * (len(word_tokens) - 1))
            punc_labels.append([int(punc)] + [int(punc)] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            cap_labels = cap_labels[: (max_seq_len - special_tokens_count)]
            punc_labels = punc_labels[: (max_seq_len - special_tokens_count)]

        
        # Add [SEP] token
        tokens += [sep_token]
        cap_labels += [pad_token_label_id] # Add special label
        punc_labels += [pad_token_label_id]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        cap_labels = [pad_token_label_id] + cap_labels
        punc_labels = [pad_token_label_id] + punc_labels

        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        cap_labels = cap_labels + ([pad_token_label_id] * padding_length)
        punc_labels = punc_labels + ([pad_token_label_id] * padding_length)
    
        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      cap_label=cap_labels,
                                      punc_label=punc_labels))
    
    return features


def load_and_cache_examples(args, tokenizer, mode):
    processors = CapPuncProcessor(args)

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

        if mode in ['train', 'dev', 'test']:
            examples = processors.get_examples(mode)
        else:
            raise Exception("For mode, Only train, dev, test is available")
        
        features = convert_examples_to_features(
            examples,
            args.max_seq_len,
            tokenizer
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
     
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_cap_label = torch.tensor([f.cap_label for f in features], dtype=torch.long)
    all_punc_label = torch.tensor([f.punc_label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_token_type_ids,
        all_attention_mask,
        all_cap_label,
        all_punc_label
    )

    return dataset