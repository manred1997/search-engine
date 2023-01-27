import logging
import os
import random

import torch
from src.end_to_end.loader.data_loader import Example, Features, Processor
from src.spelling_error.component import (
    get_homophone_letter_error,
    get_homophone_single_word_error,
    get_possible_word_from_accent_error,
    get_possible_word_from_edit_error,
    get_possible_word_from_telex_error,
    get_possible_word_from_vni_error,
)
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InputExample(Example):
    """
    A single training/test example for simple sequence labeling.
    """

    def __init__(
        self, guid, incorrect_text, correct_text, label_error_detection=None
    ) -> None:

        self.guid = guid
        self.incorrect_text = incorrect_text
        self.correct_text = correct_text
        self.label_error_detection = label_error_detection


class RobertaProcessor(Processor):
    """Processor for the data set"""

    def __init__(self, args):
        self.args = args

    def _create_examples(self, texts, labels, is_self_supervised_learning, set_type):
        """Creates examples for the training and dev sets."""

        assert len(texts) == len(labels)

        examples = []
        for i, (text, corr_text) in tqdm(enumerate(zip(texts, labels))):
            guid = "%s - %s" % (set_type, i)
            words = text.split()
            label_error_detection = [0] * len(words)
            # 1. Input process
            if is_self_supervised_learning:
                # Add noise to text
                if random.gauss(0, 1) < 2:  # one standart deviation

                    # sample error: typo/ortho graphic erros:
                    for idx, (word, label) in enumerate(
                        zip(words, label_error_detection)
                    ):
                        if (
                            random.gauss(0, 1) > self.args.prop_adding_noise
                        ):  # Probability for adding noise

                            type_spell_error = random.choices(
                                self.args.type_spell_error,
                                weights=self.args.weigth_spell_error,
                                k=1,
                            )[0]
                            if type_spell_error == "telex":
                                word_noise = random.choice(
                                    get_possible_word_from_telex_error(word)
                                )
                            elif type_spell_error == "vni":
                                word_noise = random.choice(
                                    get_possible_word_from_vni_error(word)
                                )
                            elif type_spell_error == "edit":
                                word_noise = random.choice(
                                    get_possible_word_from_edit_error(word)
                                )
                            elif type_spell_error == "accent":
                                word_noise = random.choice(
                                    get_possible_word_from_accent_error(word)
                                )
                            elif type_spell_error == "homophone_letter":
                                word_noise = random.choice(
                                    get_homophone_letter_error(word)
                                )
                            elif type_spell_error == "homophone_single_word":
                                word_noise = random.choice(
                                    get_homophone_single_word_error(word)
                                )
                            elif type_spell_error == "homophone_double_word":
                                pass  # todo
                            else:
                                raise Exception(
                                    "For type spell error, Only typographic (e.g: telex, vni, edit, ...) and orthographic type is available"
                                )

                            if word_noise != word:
                                words[idx] = word_noise
                                label_error_detection[idx] = 1
                        else:
                            continue

            text = " ".join(words)

            # elif type_spell_error == 'miss_space':
            #     text = text.split()
            #     if len(text) > 1:
            #         index_space_error = random.randint(1, len(text)-1)
            #         pairwords = " ".join(text[index_space_error-1:index_space_error+1])
            #         word_error = get_possible_word_from_miss_space_error(pairwords)
            #         text = text[:max(0, index_space_error-1)] + [word_error] + text[index_space_error+1:]
            #     text = [" ".join(text)]
            # elif type_spell_error == 'split_error':
            #     text = text.split()
            #     index_word_error = random.randint(0, len(text)-1)
            #     word_error = get_possible_word_from_split_error(text[index_word_error])
            #     text = text[:index_word_error] + [word_error] + text[index_word_error+1:]
            #     text = [" ".join(text)]
            #                     text = random.choice(text)

            examples.append(
                InputExample(
                    guid=guid,
                    incorrect_text=text,
                    correct_text=corr_text,
                    label_error_detection=label_error_detection,
                )
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, mode, self.args.token_level)
        logger.info("Looking at {}".format(data_path))

        texts_path = os.path.join(data_path, self.args.texts_file)

        if not os.path.isfile(os.path.join(data_path, self.args.correct_texts_file)):
            correct_texts_path = texts_path
        else:
            correct_texts_path = os.path.join(data_path, self.args.correct_texts_file)

        return self._create_examples(
            texts=self._read_file(texts_path),
            labels=self._read_file(correct_texts_path),
            is_self_supervised_learning=self.args.is_self_supervised_learning,
            set_type=mode,
        )


processors = RobertaProcessor


class InputFeatures(Features):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        input_sentence,
        label_ids,
        label_ids_length,
        label_length,
        label_sentence,
    ) -> None:

        # Source
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_sentence = input_sentence

        # Target
        self.label_ids = label_ids
        self.label_ids_length = label_ids_length
        self.label_length = label_length
        self.label_sentence = label_sentence


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    pad_token_label_id=-100,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_tokens = example.incorrect_text.split()
        input_sentence = " ".join(text_tokens)
        # Tokenize word by word
        tokens = []
        for word in text_tokens:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(
            len(input_ids), max_seq_len
        )
        assert (
            len(attention_mask) == max_seq_len
        ), "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert (
            len(token_type_ids) == max_seq_len
        ), "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("noise sentence: %s" % input_sentence)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids])
            )

        label_tokens = example.correct_text.split()
        label_sentence = " ".join(label_tokens)
        label_length = len(label_tokens)
        # Tokenize word by word
        tokens = []
        for word in label_tokens:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]

        label_ids_length = len(tokens)

        # Add [SEP] token
        tokens += [sep_token]

        # Add [CLS] token
        tokens = [cls_token] + tokens

        label_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_seq_len - len(label_ids)
        label_ids = label_ids + ([pad_token_label_id] * padding_length)

        assert len(label_ids) == max_seq_len, "Error with input length {} vs {}".format(
            len(label_ids), max_seq_len
        )

        if ex_index < 10:
            logger.info("clean sentence: %s" % label_sentence)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("no. sub tokens of label: %s" % str(label_ids_length))
            logger.info("no. tokens of label: %s" % str(label_length))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                input_sentence=input_sentence,
                label_ids=label_ids,
                label_ids_length=label_ids_length,
                label_length=label_length,
                label_sentence=label_sentence,
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):

    processor = processors(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode,
            args.token_level,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(
            examples=examples,
            max_seq_len=args.max_seq_len,
            tokenizer=tokenizer,
            pad_token_label_id=pad_token_label_id,
        )

        if args.save_features:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    all_input_sentences = [f.input_sentence for f in features]

    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_label_ids_length = torch.tensor(
        [f.label_ids_length for f in features], dtype=torch.int
    )
    all_label_length = torch.tensor([f.label_length for f in features], dtype=torch.int)
    all_label_sentences = [f.label_sentence for f in features]

    # dataset = TensorDataset(
    #     all_input_ids,
    #     all_attention_mask,
    #     all_token_type_ids,
    #     all_input_sentences,
    #     all_label_ids,
    #     all_label_ids_length,
    #     all_label_length,
    #     all_label_sentences
    # )
    dataset = SpellingCorrectDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_input_sentences,
        all_label_ids,
        all_label_ids_length,
        all_label_length,
        all_label_sentences,
    )
    return dataset


class SpellingCorrectDataset(Dataset):
    def __init__(
        self,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_input_sentences,
        all_label_ids,
        all_label_ids_length,
        all_label_length,
        all_label_sentences,
    ) -> None:

        self.all_input_ids = all_input_ids
        self.all_attention_mask = all_attention_mask
        self.all_token_type_ids = all_token_type_ids
        self.all_input_sentences = all_input_sentences

        self.all_label_ids = all_label_ids
        self.all_label_ids_length = all_label_ids_length
        self.all_label_length = all_label_length
        self.all_label_sentences = all_label_sentences

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, index):

        return (
            self.all_input_ids[index],
            self.all_attention_mask[index],
            self.all_token_type_ids[index],
            self.all_input_sentences[index],
            self.all_label_ids[index],
            self.all_label_ids_length[index],
            self.all_label_length[index],
            self.all_label_sentences[index],
        )
