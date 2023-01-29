import logging
import os

import numpy as np
import torch
from src.end_to_end.early_stopping import EarlyStopping
from src.end_to_end.model.loss import LabelSmoothingLoss
from src.end_to_end.trainer.trainer import Trainer
from src.utils.metrics import get_character_error_rate, get_sentence_accuracy
from src.utils.utils import MODEL_CLASSES
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class RNNTrainer(Trainer):
    def __init__(
        self,
        args,
        tokenizer=None,
        train_dataset=None,
        dev_dataset=None,
        test_dataset=None,
    ):

        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        if args.pretrained:
            self.model = self.model_class(
                input_dim=args.vocab_size,
                output_dim=args.vocab_size,
                encoder_emb_dim=args.encoder_emb_dim,
                decoder_emb_dim=args.decoder_emb_dim,
                encoder_hid_dim=args.encoder_hid_dim,
                decoder_hid_dim=args.decoder_hid_dim,
                encoder_dropout=args.encoder_dropout,
                decoder_dropout=args.decoder_dropout,
                attention=args.use_attention,
                encoder_num_layers=args.encoder_num_layers,
                encoder_bidirectional=args.encoder_bidirectional,
            )
            self.model.load_state_dict(torch.load(args.pretrained_path))
        else:
            self.model = self.model_class(
                input_dim=args.vocab_size,
                output_dim=args.vocab_size,
                encoder_emb_dim=args.encoder_emb_dim,
                decoder_emb_dim=args.decoder_emb_dim,
                encoder_hid_dim=args.encoder_hid_dim,
                decoder_hid_dim=args.decoder_hid_dim,
                encoder_dropout=args.encoder_dropout,
                decoder_dropout=args.decoder_dropout,
                attention=args.use_attention,
                encoder_num_layers=args.encoder_num_layers,
                encoder_bidirectional=args.encoder_bidirectional,
            )
        print(self.model)
        # GPU or CPU or MPS
        # torch.cuda.set_device(self.args.gpu_id)
        # logger.info('GPU ID :',self.args.gpu_id)
        logger.info(f"Target device is: {args.device}")
        self.device = args.device

        self.model.to(self.device)
        model_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"No.Params of model: {str(model_parameters)}",
        )

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
        )

        writer = SummaryWriter(log_dir=self.args.model_dir)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps
                // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc="Iteration", position=0, leave=True
            )
            logger.info("Epoch", _)

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                # batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                    "src": batch[0].to(self.device),
                    "trg": batch[3].to(self.device),
                }
                outputs = self.model(**inputs)  # B x L x Vocab_size

                outputs = outputs.to(self.device).view(
                    -1, outputs.size(2)
                )  # flatten(0, 1)
                targets = batch[3].to(self.device).transpose(0, 1).reshape(-1)

                loss_fct = LabelSmoothingLoss(
                    self.args.vocab_size,
                    padding_idx=self.tokenizer.pad_id,
                    smoothing=self.args.smoothing_cof,
                )
                loss_fct.to(self.device)
                loss = loss_fct(outputs, targets)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.args.logging_steps > 0
                        and global_step % self.args.logging_steps == 0
                    ):
                        print("\nTuning metrics:", self.args.tuning_metric)
                        results = self.evaluate("dev")
                        for metric, value in results.items():
                            writer.add_scalar(f"{metric}", value, _)
                        early_stopping(
                            results[self.args.tuning_metric], self.model, self.args
                        )
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break

                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break
            writer.add_scalar("Loss/train", tr_loss / global_step, _)

        return global_step, tr_loss / global_step

    def write_evaluation_result(self, out_file, results):
        out_file = self.args.model_dir + "/" + out_file
        w = open(out_file, "a", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        eval_loss = 0.0

        eval_character_error_rate = []

        eval_sentence_accuracy = []

        nb_eval_steps = 0

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "src": batch[0].to(self.device),
                    "trg": batch[3].to(self.device),
                    "teacher_forcing_ratio": 0,
                }
                outputs = self.model(**inputs)  # B x L x Vocab_size

                outputs = outputs.to(self.device).view(
                    -1, outputs.size(2)
                )  # flatten(0, 1)
                targets = batch[3].to(self.device).transpose(0, 1).reshape(-1)

                loss_fct = LabelSmoothingLoss(
                    self.args.vocab_size,
                    padding_idx=self.tokenizer.pad_id,
                    smoothing=self.args.smoothing_cof,
                )
                loss_fct.to(self.device)

                loss = loss_fct(outputs, targets)

                eval_loss += loss.item()

                #
                pred_ids, _ = self.predict_batch(batch[0].to(self.device))
                pred_sentences = self.tokenizer.convert_batch_ids_to_batch_sentence(
                    pred_ids
                )

                label_sentences = batch[5]

                eval_character_error_rate.extend(
                    get_character_error_rate(
                        preds=pred_sentences, targets=label_sentences
                    )[1]
                )

                eval_sentence_accuracy.extend(
                    get_sentence_accuracy(
                        preds=pred_sentences, targets=label_sentences
                    )[1]
                )

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        eval_character_error_rate = round(
            sum(eval_character_error_rate) / len(eval_character_error_rate), 4
        )

        eval_sentence_accuracy = round(
            sum(eval_sentence_accuracy) / len(eval_sentence_accuracy), 4
        )

        results = {
            "loss": eval_loss,
            "cer": eval_character_error_rate,
            "sent_acc": eval_sentence_accuracy,
        }

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if mode == "test":
            self.write_evaluation_result("eval_test_results.txt", results)
        elif mode == "dev":
            self.write_evaluation_result("eval_dev_results.txt", results)
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        torch.save(
            model_to_save.state_dict(),
            os.path.join(self.args.model_dir, "model_weights.pth"),
        )

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.model_dir, "model_weights.pth"))
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")

    def predict_batch(self, src):
        """
        src: Batch x Length

        outputs: pred_ids: Batch x Length* , pred_props: Batch x Length*
        """

        batch_size = src.shape[0]
        max_length = src.shape[1]

        pred_ids = [[self.tokenizer.sos_id] * batch_size]
        pred_props = [[1.0] * batch_size]

        time_step = 0

        self.model.eval()
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.model.forward_encoder(src)
            decoder_hidden = encoder_hidden

            while (
                not all(np.any(np.asarray(pred_ids).T == self.tokenizer.eos_id, axis=1))
                and time_step <= max_length + 10
            ):
                trg = torch.tensor(pred_ids).to(self.device)

                decoder_output, (
                    decoder_hidden,
                    encoder_outputs,
                ) = self.model.forward_decoder(trg, decoder_hidden, encoder_outputs)

                decoder_output = decoder_output.softmax(dim=-1)

                props, indices = torch.topk(decoder_output, k=1)

                props = props.flatten().tolist()
                indices = indices.flatten().tolist()

                pred_props.append(props)
                pred_ids.append(indices)

                time_step += 1

            pred_ids = np.asarray(pred_ids).T

            pred_props = np.asarray(pred_props).T

            # pred_props = np.multiply(pred_props, pred_ids > 3)
            # pred_props = np.sum(pred_props, axis=-1) / (pred_props > 0).sum(-1)

        return pred_ids, pred_props
