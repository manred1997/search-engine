from typing import List

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import (
    AlbertTokenizerFast,
    AutoTokenizer,
    BertTokenizerFast,
    GPT2TokenizerFast,
)


def train_tokenizer_from_exiting_tokenizer(
    corpus: List,
    batch_size: int,
    model_name_or_path: str,
    vocab_size: int,
    save_tokenizer=True,
):

    """
    This function just use the `train_new_from_iterator` API for train existing tokenizer.
    """

    def batch_iterator():
        for i in range(0, len(corpus), batch_size):
            yield corpus[i : i + batch_size]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.is_fast:
        print("The Tokenizer based on the Rust library tokenizers")
    else:
        print("The Tokenizer based on a full python implementation")
    try:
        new_tokenizer = tokenizer.train_new_from_iterator(
            batch_iterator(), vocab_size=vocab_size
        )

    except AttributeError:
        print("The init Tokenizer not have train_new_from_iterator API")
        return
    if save_tokenizer:
        model_name_or_path = "owner-" + model_name_or_path.split("/")[-1] + "-tokenizer"
        new_tokenizer.save_pretrained(model_name_or_path)
        print(f"Tokenizer is saved at {model_name_or_path}")
    return


def train_tokenizer_from_scratch(
    corpus: List,
    tokenization_algorithm: str,
    normarlizers: List[str],
    vocab_size: int,
    batch_size: int,
    save_tokenizer=True,
    **kwargs,
):
    """
    This function takes several steps:

        1. Normalization: Executes all the initial transformations over the initial input string. For example when you need to lowercase some text, maybe strip it, or even apply one of the common unicode normalization process, you will add a Normalizer

        2. Pre-tokenization: In charge of splitting the initial input string. That's the component that decides where and how to pre-segment the origin string. The simplest example would be to simply split on spaces

        3. Model: Handles all the sub-token discovery and generation, this is the part that is trainable and really dependent of your input data

        4. Post-Processing: Provides advanced construction features to be compatible with some of the Transformers-based SoTA models. For instance, for BERT it would wrap the tokenized sentence around [CLS] and [SEP] tokens
    """

    def batch_iterator():
        for i in range(0, len(corpus), batch_size):
            yield corpus[i : i + batch_size]

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    # Select tokenization algorithm
    if tokenization_algorithm == "WordPiece":
        model = models.WordPiece(unk_token=special_tokens[0])
    elif tokenization_algorithm == "BPE":
        model = models.BPE()
    elif tokenization_algorithm == "Unigram":
        model = models.Unigram()
    else:
        raise Exception("For tokenization algorithm, only WordPiece, BPE, Unigram")

    tokenizer = Tokenizer(model)

    # Select Normalizer
    normalizer_items = []
    for normalizer in normarlizers:
        if normalizer == "NFD":
            normalizer_items.append(normalizers.NFD())
        elif normalizer == "NFKD":
            normalizer_items.append(normalizers.NFKD())
        elif normalizer == "NFC":
            normalizer_items.append(normalizers.NFC())
        elif normalizer == "NFKC":
            normalizer_items.append(normalizers.NFKC())
        elif normalizer == "lowercase":
            normalizer_items.append(normalizers.Lowercase())
        elif normalizer == "strip":
            normalizer_items.append(normalizers.Strip())
        elif normalizer == "stripaccents":
            normalizer_items.append(normalizers.StripAccents())
        elif normalizer == "nmt":
            normalizer_items.append(normalizers.Nmt())
        elif normalizer == "bertnormalizer":
            normalizer_items.append(normalizers.BertNormalizer())
        else:
            raise Exception(
                "For normalizer item name, only [NFD, NFKD, NFC, NFKC, lowercase, strip, stripaccents, nmt]"
            )

    if not normarlizers:
        # Default normalizer
        normalizer_items.append(normalizers.Strip())
        normalizer_items.append(normalizers.NFD())
    # Expect normalizer
    normalizer_items.append(normalizers.Replace("``", '"'))
    normalizer_items.append(normalizers.Replace("''", '"'))

    tokenizer.normalizer = normalizers.Sequence(normalizer_items)

    # Select Pre-tokenizer
    if tokenization_algorithm == "WordPiece":
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    elif tokenization_algorithm == "BPE":
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    elif tokenization_algorithm == "Unigram":
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    else:
        raise Exception("For tokenization algorithm, only WordPiece, BPE, Unigram")

    # Trainer
    if tokenization_algorithm == "WordPiece":
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens
        )
    elif tokenization_algorithm == "BPE":
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=["<|endoftext|>"]
        )
    elif tokenization_algorithm == "Unigram":
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"],
            unk_token="<unk>",
        )
    else:
        raise Exception("For tokenization algorithm, only WordPiece, BPE, Unigram")

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    # Post-Process
    if tokenization_algorithm == "WordPiece":
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_token_id),
                ("[SEP]", sep_token_id),
            ],
        )
    elif tokenization_algorithm == "BPE":
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    elif tokenization_algorithm == "Unigram":
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_token_id),
                ("[SEP]", sep_token_id),
            ],
        )
    else:
        raise Exception("For tokenization algorithm, only WordPiece, BPE, Unigram")

    # Decoder
    if tokenization_algorithm == "WordPiece":
        tokenizer.decoder = decoders.WordPiece(prefix="##")
        tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
    elif tokenization_algorithm == "BPE":
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    elif tokenization_algorithm == "Unigram":
        tokenizer.decoder = decoders.Metaspace()
        tokenizer = AlbertTokenizerFast(tokenizer_object=tokenizer)
    else:
        raise Exception("For tokenization algorithm, only WordPiece, BPE, Unigram")

    if save_tokenizer:
        model_name_or_path = "owner-" + tokenization_algorithm + "-tokenizer"
        tokenizer.save_pretrained(model_name_or_path)
        print(f"Tokenizer is saved at {model_name_or_path}")
