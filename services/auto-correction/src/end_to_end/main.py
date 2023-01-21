import os
import sys
import torch
import argparse

sys.path.append(os.environ.get('PROJECT_PATH'))

from src.end_to_end.loader.roberta_data_loader import load_and_cache_examples
from src.end_to_end.trainer.roberta_trainer import RobertaTrainer
from src.utils.utils import MODEL_CLASSES, MODEL_PATH_MAP
from src.utils.utils import init_logger, load_tokenizer, set_seed


def main(args):

    init_logger()
    set_seed(args.seed)
    tokenizer = load_tokenizer(args)

    args.vocab_size = tokenizer.vocab_size

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = RobertaTrainer(args, tokenizer, train_dataset, dev_dataset, test_dataset)

    if args.do_eval:
        trainer.evaluate("dev")
        # trainer.evaluate("test")

    
    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")
        trainer.evaluate("test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    
    ## Config process data
    parser.add_argument("--is_self_supervised_learning", action="store_true", help="Self-supervised learning")
    parser.add_argument("--type_spell_error", default=['telex', 'vni', 'edit', 'accent', 'homophone_letter', 'homophone_single_word'], type=list, help="Types of spelling error")
    parser.add_argument("--weigth_spell_error", default=[20, 5, 65, 10, 15, 15], type=list, help="Weighting types of spelling error")
    parser.add_argument("--prop_adding_noise", default=1, type=int, help="Probability of adding noise")


    ## Resource path   
    parser.add_argument("--data_dir", default="resources/auto-correction/data", type=str, help="The input data dir")
    parser.add_argument("--texts_file", default="corpus.txt", type=str, help="The input data file")
    parser.add_argument("--correct_texts_file", default="correct_corpus.txt", type=str, help="The output data file")


    ## Type Model
    parser.add_argument("--model_dir", default="Spell-Check", required=True, type=str, help="Path to save, load model")
    parser.add_argument(
        "--model_type",
        default="scRoberta",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--tuning_metric", default="loss", type=str, help="Metrics to tune when training")

    ## Config-Model
    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )
    parser.add_argument(
        "--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument(
        "--ignore_index",
        default=0,
        type=int,
        help="Specifies a target value that is ignored and does not contribute to the input gradient",
    )
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--freeze_backbone", default=False, type=bool, help="Freeze Transformer layers")
    parser.add_argument("--encoder_emb_dim", default=256, type=int, help="Embedding dimension in Encoder block")
    parser.add_argument("--decoder_emb_dim", default=256, type=int, help="Embedding dimension in Decoder block")
    parser.add_argument("--encoder_hid_dim", default=512, type=int, help="Hidden state dimension of RNN module in Encoder block")
    parser.add_argument("--decoder_hid_dim", default=512, type=int, help="Hidden state dimension of RNN module in Decoder block")
    parser.add_argument("--encoder_dropout", default=0.1, type=float, help="Dropout for in Encoder block")
    parser.add_argument("--decoder_dropout", default=0.1, type=float, help="Dropout for in Decoder block")
    parser.add_argument("--use_attention", action="store_true", help="Use attention in Decoder Phase")
    parser.add_argument("--encoder_num_layers", default=2, type=int, help="Num layers RNN module in Encoder block")
    parser.add_argument("--encoder_bidirectional", default=True, type=bool, help="Use Bidirectional RNN in Encder block")
    


    ## Config-Optimizer
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")

    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size for evaluation.")
    
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )


    # Config Hardware
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # Config Logging
    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")    
    
    # Read/Write features/models
    parser.add_argument("--save_features", action="store_true", help="Save features into cached file")
    parser.add_argument("--pretrained", action="store_true", help="Whether to init model from pretrained base model")
    parser.add_argument("--pretrained_path", default="./Spell-Check", type=str, help="The pretrained model path")

    args = parser.parse_args()

    #
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    args.device = "mps" if torch.backends.mps.is_available() \
        else "cuda" if torch.cuda.is_available() else "cpu"

    main(args)