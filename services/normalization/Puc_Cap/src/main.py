import torch
import argparse

from dataloader import load_and_cache_examples
from trainer import Trainer
from utils import init_logger, set_seed, load_tokenizer, MODEL_PATH_MAP

def main(args):
    init_logger()
    set_seed(args)

    tokenizer = load_tokenizer(args)

    punc_label_set = None

    train_dataset = load_and_cache_examples(args, punc_label_set, tokenizer, 'train')
    dev_dataset = load_and_cache_examples(args, punc_label_set, tokenizer, 'dev')
    test_dataset = load_and_cache_examples(args, punc_label_set, tokenizer, 'test')

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")
        trainer.evaluate("test")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
    #                     help="Pre-trained model selected in the list: bert-base-multilingual-uncased, "
    #                          "bert-base-multilingual-cased...")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Pre-trained model type selected in the list: electra, bert, xlmr.")
                    
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--max_seq_length",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--use_cap_emb",
                        action='store_true',
                        help="Whether to use capitalization embedding")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--tuning_metric", default="mean_f1", type=str, help="Metrics to tune when training")

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")

    parser.add_argument('--loss_coef',default=0.5, type=float)


    parser.add_argument('--cap_emb_dim',default=100,type=int)
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument('--num_punc', default=None, type=int)
    parser.add_argument('--num_cap',default=3, type=int)
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=5,
        help="Number of unincreased validation step to wait for early stopping",
    )


    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    main(args)