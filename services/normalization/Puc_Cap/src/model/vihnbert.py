import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

from .module import FeedforwardLayer

class ViHnBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, cap_label_list, punc_label_list, freeze_bert=False) -> None:
        super(ViHnBERT, self).__init__(config)

        self.args = args
        
        self.num_cap_label = len(cap_label_list)
        self.num_punc_label = len(punc_label_list)

        self.roberta = RobertaModel(config)  # Load pretrained bert
        if freeze_bert:
            for p in self.roberta.parameters():
                p.requires_grad = False

        self.cap_layer = FeedforwardLayer(config.hidden_size, self.num_cap_label)

        if self.args.use_cap_emb:
            self.punc_embed_matrix = nn.Parameter(
                torch.rand(self.num_cap_label, args.args.cap_emb_dim), requires_grad=True
            )
            self.punc_layer = FeedforwardLayer(config.hidden_size + self.args.cap_emb_dim, self.num_punc_label)
        else:
            self.punc_layer = FeedforwardLayer(config.hidden_size, self.num_punc_label) 

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_punc_label, batch_first=True)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        cap_labels = None,
        pun_labels = None
    ):

        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0] # Batchsize X Seq_len X hidden_size


        cap_logits = self.cap_layer(sequence_output)

        if self.args.use_cap_emb:
        # compute cap embedding

            cap_dis = F.softmax(cap_logits, dim=-1)

            punc_embed_matrix_dup = self.punc_embed_matrix.repeat(cap_dis.size(0), 1, 1)
            punc_emb = torch.matmul(cap_dis, punc_embed_matrix_dup)
            input_punc = torch.cat([sequence_output, punc_emb],dim = -1)
        else:
            input_punc = sequence_output

        pun_logits = self.punc_layer(input_punc)

        total_loss = 0
        if cap_labels is not None:

            cap_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            cap_loss = cap_loss_fct(
                cap_logits.view(-1, self.num_cap_label), cap_labels.view(-1)
            )
            total_loss += self.args.loss_coef * cap_loss
        
        if pun_labels is not None:

            if self.args.use_crf:
                pun_loss = self.crf(pun_logits, pun_labels, mask=attention_mask.byte(), reduction="mean")
                pun_loss = -1 * pun_loss  # negative log-likelihood

            else:
                pun_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = pun_logits.view(-1, self.num_punc_label)[active_loss]
                    active_labels = pun_labels.view(-1)[active_loss]
                    pun_loss = pun_loss_fct(active_logits, active_labels)
                else:
                    pun_loss = pun_loss_fct(pun_logits.view(-1, self.num_punc_label), pun_labels.view(-1))
            total_loss += (1 - self.args.loss_coef) * pun_loss

        
        outputs = ((cap_logits, pun_logits),)

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits