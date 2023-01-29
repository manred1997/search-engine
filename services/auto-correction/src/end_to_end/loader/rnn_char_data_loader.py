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
from src.utils.constants import AVAILABLE_CHARACTER
from torch.nn.utils.rnn import pad_sequence
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


class InputFeatures(Features):
    def __init__(
        self,
        input_ids,
        attention_mask,
        input_sentence,
        label_ids=None,
        label_length=None,
        label_sentence=None,
    ) -> None:

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.input_sentence = input_sentence
        self.label_ids = label_ids
        self.label_length = label_length
        self.label_sentence = label_sentence


class RNNCharProcessor(Processor):
    """Processor for the data set"""

    def __init__(self, args) -> None:
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


class CharacterTokenizer(object):
    def __init__(self) -> None:

        self.pad_char = "<pad>"
        self.sos_char = "<sos>"
        self.eos_char = "<eos>"
        self.unk_char = "<unk>"

        self.available_character = list(AVAILABLE_CHARACTER)

        self.available_character = [self.unk_char] + self.available_character
        self.available_character = [self.eos_char] + self.available_character
        self.available_character = [self.sos_char] + self.available_character
        self.available_character = [self.pad_char] + self.available_character

        self.char_to_id = {}
        self.id_to_char = {}

        for idx, char in enumerate(self.available_character):
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

        self.pad_id = self.char_to_id[self.pad_char]
        self.sos_id = self.char_to_id[self.sos_char]
        self.eos_id = self.char_to_id[self.eos_char]
        self.unk_id = self.char_to_id[self.unk_char]

        self.vocab_size = self.__len_vocab__()

    def __len_vocab__(self):
        return len(self.char_to_id)

    def convert_token_to_char_ids(self, word):
        return [self.char_to_id[char] for char in word]

    def convert_char_ids_to_token(self, ids):
        return [self.id_to_char[id] for id in ids]

    def convert_char_to_id(self, char):
        return self.char_to_id[char]

    def convert_id_to_char(self, id):
        return self.id_to_char[id]

    def _get_vocab_len(self):
        return len(self.char_to_id)

    def convert_ids_to_sentence(self, ids):
        start_indice = 1 if self.sos_id in ids else 0
        end_indice = ids.index(self.eos_id) if self.eos_id in ids else None
        ids = ids[start_indice:end_indice]
        sentence = "".join([self.convert_id_to_char(id) for id in ids])
        return sentence, ids

    def convert_batch_ids_to_batch_sentence(self, batch_ids):
        sentences = []
        for ids in batch_ids:
            sentence, _ = self.convert_ids_to_sentence(ids)
            sentences.append(sentence)
        return sentences


class Seq2SeqTokenizer:
    @classmethod
    def from_pretrained(cls, path):
        # return custom tokenizer follow path
        return CharacterTokenizer()


processors = RNNCharProcessor


def convert_examples_to_features(
    examples, max_seq_tokens, tokenizer, mask_padding_with_zero=True
):

    sos_char = tokenizer.sos_char
    eos_char = tokenizer.eos_char
    unk_char = tokenizer.unk_char

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_tokens = example.incorrect_text.split()[:max_seq_tokens]
        input_sentence = " ".join(text_tokens)
        # Tokenize word by word
        input_ids = []
        for char in input_sentence:
            input_ids.append(
                tokenizer.convert_char_to_id(char)
                if char in tokenizer.char_to_id
                else tokenizer.convert_char_to_id(unk_char)
            )

        # Add end of sentence id
        input_ids += [tokenizer.convert_char_to_id(eos_char)]
        # Add start of sentence id
        input_ids = [tokenizer.convert_char_to_id(sos_char)] + input_ids

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("noise sentence: %s" % input_sentence)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )

        label_tokens = example.correct_text.split()[:max_seq_tokens]
        label_sentence = " ".join(label_tokens)
        label_length = len(label_tokens)

        # Tokenize word by word
        label_ids = []
        for char in label_sentence:
            label_ids.append(
                tokenizer.convert_char_to_id(char)
                if char in tokenizer.char_to_id
                else tokenizer.convert_char_to_id(unk_char)
            )

        # Add end of sentence id
        label_ids += [tokenizer.convert_char_to_id(eos_char)]
        # Add start of sentence id
        label_ids = [tokenizer.convert_char_to_id(sos_char)] + label_ids

        if ex_index < 5:
            logger.info("clean sentence: %s" % label_sentence)
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_sentence=input_sentence,
                label_ids=label_ids,
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

        print(len(examples))
        features = convert_examples_to_features(
            examples=examples, max_seq_tokens=args.max_seq_len, tokenizer=tokenizer
        )
        if args.save_features:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    pad_token_label_id = args.ignore_index
    # Convert to Tensors and build dataset

    all_input_ids = [torch.tensor(f.input_ids) for f in features]
    all_input_ids = pad_sequence(
        all_input_ids,
        batch_first=True,
        padding_value=tokenizer.convert_char_to_id(tokenizer.pad_char),
    )
    all_attention_mask = [torch.tensor(f.attention_mask) for f in features]
    all_attention_mask = pad_sequence(
        all_attention_mask, batch_first=True, padding_value=pad_token_label_id
    )
    all_input_sentences = [f.input_sentence for f in features]

    all_label_ids = [torch.tensor(f.label_ids) for f in features]
    all_label_ids = pad_sequence(
        all_label_ids,
        batch_first=True,
        padding_value=tokenizer.convert_char_to_id(tokenizer.pad_char),
    )
    all_label_length = [f.label_length for f in features]
    all_label_sentences = [f.label_sentence for f in features]

    assert all_input_ids.size() == all_attention_mask.size()
    assert (
        len(all_input_ids)
        == len(all_input_sentences)
        == len(all_label_ids)
        == len(all_label_length)
        == len(all_label_sentences)
    )

    dataset = SpellingCorrectDataset(
        all_input_ids=all_input_ids,
        all_attention_mask=all_attention_mask,
        all_input_sentences=all_input_sentences,
        all_label_ids=all_label_ids,
        all_label_length=all_label_length,
        all_label_sentences=all_label_sentences,
    )
    return dataset


class SpellingCorrectDataset(Dataset):
    def __init__(
        self,
        all_input_ids,
        all_attention_mask,
        all_input_sentences,
        all_label_ids,
        all_label_length,
        all_label_sentences,
    ) -> None:

        self.all_input_ids = all_input_ids
        self.all_attention_mask = all_attention_mask
        self.all_input_sentences = all_input_sentences

        self.all_label_ids = all_label_ids
        self.all_label_length = all_label_length
        self.all_label_sentences = all_label_sentences

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, index):

        return (
            self.all_input_ids[index],
            self.all_attention_mask[index],
            self.all_input_sentences[index],
            self.all_label_ids[index],
            self.all_label_length[index],
            self.all_label_sentences[index],
        )
