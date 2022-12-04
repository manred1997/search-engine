from __future__ import absolute_import, division, print_function

import logging

import torch
from torch import nn
from model.feedforward import FeedforwardLayer
from torchcrf import CRF
import torch.nn.functional as F
from transformers import RobertaPreTrainedModel, XLMRobertaModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class ViCapPuncXLMRModel(RobertaPreTrainedModel):

    def __init__(self,config, args) -> None:
        super().__init__(config)
        self.num_punc = args.num_punc
        self.num_cap = args.num_cap
        self.args = args

        self.electra = XLMRobertaModel(config)
        self.cap_layer = FeedforwardLayer(config.hidden_size , self.num_cap)
        if args.use_cap_emb:
            self.punc_embed_matrix = nn.Parameter(
                torch.rand(self.num_cap, args.cap_emb_dim), requires_grad=True
            )
            self.punc_layer = FeedforwardLayer(config.hidden_size + args.cap_emb_dim, self.num_punc)
        else:
            self.punc_layer = FeedforwardLayer(config.hidden_size, self.num_punc) 


        if args.use_crf:
            self.crf = CRF(num_tags=self.num_punc, batch_first=True)
    

    def forward(
            self,
            input_ids=None,
            words_lengths=None,
            attention_mask=None,
            attention_mask_label=None,
            cap_label = None,
            punc_label = None
    ):


        outputs = self.electra(input_ids,attention_mask=attention_mask)

        context_embedding = outputs[0]

        # Compute align word sub_word matrix
        batch_size = input_ids.shape[0]
        max_sub_word = input_ids.shape[1]
        max_word = words_lengths.shape[1]
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))

        for i, sample_length in enumerate(words_lengths):
            for j in range(len(sample_length)):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][start_idx: start_idx + sample_length[j]] = 1 if sample_length[j] > 0 else 0

        align_matrix = align_matrix.to(context_embedding.device)
        # Combine sub_word features to make word feature
        context_embedding_align = torch.bmm(align_matrix, context_embedding)

        cap_logits = self.cap_layer(context_embedding_align)

        if self.args.use_cap_emb:
        # compute cap embedding

            cap_dis = F.softmax(cap_logits, dim=-1)

            punc_embed_matrix_dup = self.punc_embed_matrix.repeat(cap_dis.size(0), 1, 1)
            punc_emb = torch.matmul(cap_dis, punc_embed_matrix_dup)


            input_punc = torch.cat([context_embedding_align, punc_emb],dim = -1)
        else:
            input_punc = context_embedding_align


        punc_logits = self.punc_layer(input_punc)

        total_loss = 0
        if cap_label is not None:

            cap_loss_fct = nn.CrossEntropyLoss()
            cap_loss = cap_loss_fct(
                cap_logits.view(-1, self.num_cap), cap_label.view(-1)
            )
            total_loss += self.args.loss_coef * cap_loss
        
        if punc_label is not None:

            if self.args.use_crf:
                punc_loss = self.crf(punc_logits, punc_label, mask=attention_mask_label.byte(), reduction="mean")
                punc_loss = -1 * punc_loss  # negative log-likelihood

            else:
                punc_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                # Only keep active parts of the loss
                # attention_mask_label = None
                if attention_mask_label is not None:
                    active_loss = attention_mask_label.view(-1) == 1
                    active_logits = punc_logits.view(-1, self.num_punc)[active_loss]
                    active_labels = punc_label.view(-1)[active_loss]
                    punc_loss = punc_loss_fct(active_logits, active_labels)
                else:
                    punc_loss = punc_loss_fct(punc_logits.view(-1, self.num_punc), punc_label.view(-1))
            total_loss += (1 - self.args.loss_coef) * punc_loss

        outputs = ((cap_logits, punc_logits),)

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits



 




