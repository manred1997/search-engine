import logging
import os

import numpy as np
import torch
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import MODEL_CLASSES, get_pun_labels, get_cap_labels, compute_metrics



logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset, dev_dataset, test_dataset) -> None:
        self.args = args

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.cap_label_lst = get_cap_labels(args)
        self.pun_label_lst = get_pun_labels(args)

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        if args.pretrained:
            self.config = self.config_class.from_pretrained(args.pretrained_path, finetuning_task=args.token_level)
            self.model = self.model_class.from_pretrained(
                args.pretrained_path,
                config=self.config,
                args=args,
            )
        else:
            self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.token_level)
            self.model = self.model_class.from_pretrained(
                args.model_name_or_path,
                config=self.config,
                args=args
            )

        # GPU or CPU
        torch.cuda.set_device(self.args.gpu_id)
        print('GPU ID :',self.args.gpu_id)
        print('Cuda device:',torch.cuda.current_device())
        self.device = args.device

        self.model.to(self.device)
        model_parameters =  sum(p.numel() for p in self.model.parameters() if p.requires_grad)  
        print('#params:',model_parameters)

    def train(self):

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        writer = SummaryWriter(log_dir=self.args.model_dir)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        results = self.evaluate("dev")
        print(results)
        results = self.evaluate("test")
        print(results)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            print("\nEpoch", _)
            tr_loss = 0.0
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                    'input_ids':batch[0],
                    'attention_mask':batch[2],
                    'cap_label':batch[3],
                    'pun_label':batch[4],
                }

                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[1]

                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    print("\nTuning metrics:", self.args.tuning_metric)
                    results = self.evaluate("dev")
                    writer.add_scalar("Loss/validation", results["loss"], _)
                    writer.add_scalar("Slot F1/validation", results["slot_f1"], _)
                    early_stopping(results[self.args.tuning_metric], self.model, self.args)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

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
        w = open(out_file, "w", encoding="utf-8")
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
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        # init variable
        eval_loss = 0.0
        nb_eval_steps = 0

        cap_preds = None
        out_cap_labels = None
        pun_preds = None
        out_pun_labels = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids':batch[0],
                    'attention_mask':batch[2],
                    'cap_labels':batch[3],
                    'pun_labels':batch[4],
                }

                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[1]

                outputs = self.model(**inputs)

                tmp_eval_loss, (cap_logits, pun_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1


            # capitalization prediction
            if cap_preds is None:
                cap_preds = cap_logits.detach().cpu().numpy()
                out_cap_labels = inputs["cap_labels"].detach().cpu().numpy()
            else:
                cap_preds = np.append(cap_preds, cap_logits.detach().cpu().numpy(), axis=0)
                out_cap_labels = np.append(
                    out_cap_labels, inputs["cap_labels"].detach().cpu().numpy(), axis=0
                )

            # puntuation prediction
            if pun_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    pun_preds = np.array(self.model.crf.decode(pun_logits))
                else:
                    pun_preds = pun_logits.detach().cpu().numpy()

                out_pun_labels = inputs["pun_labels"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    pun_preds = np.append(pun_preds, np.array(self.model.crf.decode(pun_logits)), axis=0)
                else:
                    pun_preds = np.append(pun_preds, pun_logits.detach().cpu().numpy(), axis=0)

                out_pun_labels = np.append(
                    out_pun_labels, inputs["pun_labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        cap_preds = np.argmax(cap_preds, axis=2)
        if not self.args.use_crf:
            pun_preds = np.argmax(pun_preds, axis=2)

        cap_label_set = {i: label for i, label in enumerate(self.cap_label_lst)}
        pun_label_set = {i: label for i, label in enumerate(self.pun_label_lst)}
        
        out_cap_label_list = [[] for _ in range(out_cap_labels.shape[0])]
        cap_preds_list = [[] for _ in range(out_cap_labels.shape[0])]
        out_pun_label_list = [[] for _ in range(out_pun_labels.shape[0])]
        pun_preds_list = [[] for _ in range(out_pun_labels.shape[0])]

        for i in range(out_cap_labels.shape[0]):
            for j in range(out_cap_labels.shape[1]):
                if out_cap_labels[i, j] != self.pad_token_label_id:
                    out_cap_label_list[i].append(cap_label_set[out_cap_labels[i][j]])
                    cap_preds_list[i].append(cap_label_set[cap_preds[i][j]])
                if out_pun_labels[i, j] != self.pad_token_label_id:
                    out_pun_label_list[i].append(pun_label_set[out_pun_labels[i][j]])
                    pun_preds_list[i].append(pun_label_set[pun_preds[i][j]])

        total_result = compute_metrics(cap_preds_list, out_cap_label_list, out_pun_label_list, pun_preds_list, self.args.loss_coef)
        results.update(total_result)

        
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
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    
    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(
                self.args.model_dir,
                config=self.config,
                args=self.args,
                slot_label_lst=self.slot_label_lst
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")