import logging

import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel


logger = logging.getLogger(__name__)

class LayerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate=0.0) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class E2ESpellCheckRoberta(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(E2ESpellCheckRoberta, self).__init__(config)
        self.args = args

        logger.info("Load pretrained/check point model")
        self.roberta = RobertaModel(config)

        self.layer_classifier = LayerClassifier(config.hidden_size, self.args.vocab_size)

        if self.args.freeze_backbone:
            # Uncomment to freeze BERT layers
            for param in self.roberta.parameters():
                param.requires_grad = False


    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids=None,
                targets=None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0] # B x len_seq x hidden_size

        logits = self.layer_classifier(sequence_output)

        total_loss = 0

        # Check loss:
        if targets is not None:

            # logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]

            loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.args.ignore_index)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.args.vocab_size)[active_loss]
                active_labels = targets.view(-1)[active_loss]

                loss = loss_fct(active_logits, active_labels)

            else:
                loss = loss_fct(logits.view(-1, self.args.vocab_size), targets.view(-1))
            total_loss += loss

        outputs = ((logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits