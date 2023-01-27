import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, args):
        if args.tuning_metric == "loss":
            score = -val_loss
        else:
            score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, args):
        """Saves model when validation loss decreases or accuracy/f1 increases."""
        if self.verbose:
            if args.tuning_metric == "loss":
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )
            else:
                print(
                    f"{args.tuning_metric} increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )
        # model.save_pretrained(args.model_dir)
        # torch.save(args, os.path.join(args.model_dir, "training_args.bin"))
        self.val_loss_min = val_loss

        # Save model checkpoint (Overwrite)
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        model_to_save = model.module if hasattr(model, "module") else model

        try:
            model_to_save.save_pretrained(self.args.model_dir)
        except Exception:
            torch.save(
                model_to_save.state_dict(),
                os.path.join(args.model_dir, "model_weights.pth"),
            )

        # # Save training arguments together with the trained model
        torch.save(args, os.path.join(args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", args.model_dir)
