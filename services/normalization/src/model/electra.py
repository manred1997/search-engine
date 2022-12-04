from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from transformers import (ElectraForTokenClassification)
from torchcrf import CRF

from utils.punc_dataset import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PuncElectraModel(ElectraForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.electra(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class PuncElectraCrfModel(ElectraForTokenClassification):
    def __init__(self, config):
        super(PuncElectraCrfModel, self).__init__(config=config)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.electra(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
#         print(attention_mask_label)
        total_loss = 0
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=attention_mask_label.byte())
            total_loss += -1.0 * log_likelihood
        outputs = ((logits, logits),)

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits
        


class PuncElectraLstmModel(ElectraForTokenClassification):
    def __init__(self, config):
        super(PuncElectraLstmModel, self).__init__(config=config)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=2,
                            batch_first=True, bidirectional=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.electra(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        sequence_output, _ = self.lstm(sequence_output)

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class PuncElectraLstmCrfModel(ElectraForTokenClassification):
    def __init__(self, config):
        super(PuncElectraLstmCrfModel, self).__init__(config=config)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.electra(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        sequence_output, _ = self.lstm(sequence_output)

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=attention_mask_label.type(torch.uint8))
            return -1.0 * log_likelihood
        else:
            sequence_tags = self.crf.decode(logits)
            return sequence_tags
