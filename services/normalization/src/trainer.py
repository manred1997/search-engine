import logging
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm, trange
from utils.early_stopping import EarlyStopping
from utils.trainer_state import TrainerState
from utils.metrics import compute_metrics



logger = logging.getLogger(__name__)

class ViCapPuncTrainer(object):


    def __init__(self, args, model, train_dataset, dev_dataset, test_dataset) :
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer_state = TrainerState()
        self.model.to(self.device)

    def get_train_dataloader(self):
        sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=sampler, batch_size=self.args.train_batch_size)
        return train_dataloader


    def get_eval_dataloader(self, eval_dataset):
        sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,sampler=sampler, batch_size=self.args.eval_batch_size)
        return eval_dataloader


    def train(self):

        train_dataloader = self.get_train_dataloader()
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        print("check init")

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
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(self.args.warmup_proportion*t_total), num_training_steps=t_total
        )



        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)


        global_step = 0
        self.model.zero_grad()
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)


        start_epoch = int(self.trainer_state.epoch)
        self.model.train()

        for epoch in trange(start_epoch, self.args.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.args.num_train_epochs}")

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            print("\nEpoch", epoch)
            tr_loss = 0
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                    'input_ids':batch[0],
                    'words_lengths':batch[1],
                    'attention_mask':batch[2],
                    'attention_mask_label':batch[3],
                    'cap_label':batch[4],
                    'punc_label':batch[5],
                }
                
                loss = self.compute_loss(self.model,inputs)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    
                    self.trainer_state.epoch = epoch
                    self.trainer_state.global_step = global_step
                    self.trainer_state.max_steps = t_total
                    self.trainer_state.loss = tr_loss/(step+1)
                if (step +1) % self.args.logging_steps ==0:
                    logger.info('\n%s',self.trainer_state.to_string())

            results = self.evaluate('dev')
            early_stopping(results[self.args.tuning_metric], self.args)
            if early_stopping.counter == 0:
                self.save_model()
            if early_stopping.early_stop:
                print("Early stopping")
                break



    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return outputs if return_outputs else outputs[0]
    
    def evaluate(self, mode):
        if mode == "test":
            eval_dataset = self.test_dataset
        elif mode == "dev":
            eval_dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")



        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        logger.info("***** Running evaluation on %s dataset *****",mode)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        cap_preds = None
        punc_preds = None
        label_masks = None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids':batch[0],
                    'words_lengths':batch[1],
                    'attention_mask':batch[2],
                    'attention_mask_label':batch[3],
                    'cap_label':batch[4],
                    'punc_label':batch[5],
                }
                
                tmp_eval_loss, (cap_logits, punc_logits)  = self.compute_loss(self.model,inputs,return_outputs=True)
                

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # capitalization prediction
            if cap_preds is None:
                cap_preds = cap_logits.detach().cpu().numpy()
                out_cap_labels = inputs["cap_label"].detach().cpu().numpy()
            else:
                cap_preds = np.append(cap_preds, cap_logits.detach().cpu().numpy(), axis=0)
                out_cap_labels = np.append(
                    out_cap_labels, inputs["cap_label"].detach().cpu().numpy(), axis=0
                )
            

            # puntuation prediction
            if punc_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    punc_preds = np.array(self.model.crf.decode(punc_logits))
                else:
                    punc_preds = punc_logits.detach().cpu().numpy()

                out_punc_labels = inputs["punc_label"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    punc_preds = np.append(punc_preds, np.array(self.model.crf.decode(punc_logits)), axis=0)
                else:
                    punc_preds = np.append(punc_preds, punc_logits.detach().cpu().numpy(), axis=0)

                out_punc_labels = np.append(
                    out_punc_labels, inputs["punc_label"].detach().cpu().numpy(), axis=0
                )
            
            if label_masks is None:
                label_masks = batch[3].detach().cpu().numpy()
            else:
                label_masks = np.append(label_masks, batch[3].detach().cpu().numpy(), axis=0)
        
        cap_preds = np.argmax(cap_preds, axis=2)
        if not self.args.use_crf:
            punc_preds = np.argmax(punc_preds, axis=2)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}
        results.update(compute_metrics(cap_preds,out_cap_labels,punc_preds,out_punc_labels,label_masks,0.5))

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if mode == "test":
            self.write_evaluation_result("eval_test_results.txt", results)
        elif mode == "dev":
            self.write_evaluation_result("eval_dev_results.txt", results)

        return results




    def save_model(self):
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.output_dir)


        self.trainer_state.save_to_json(os.path.join(self.args.output_dir, "trainer_state.json"))

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.output_dir)



    def load_model(self,path=None):
        # Check whether model exists

        if path is None:
            path = self.args.output_dir

        if not os.path.exists(self.args.output_dir):
            raise Exception("Model doesn't exists! Train first!")
        
        args = torch.load(os.path.join(path, "training_args.bin"))
        self.model = self.model.from_pretrained(
                path,
                args=args,
            )
        
        self.model.to(self.device)


    def write_evaluation_result(self, out_file, results):
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        out_file = self.args.output_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()





