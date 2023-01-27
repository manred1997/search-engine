import torch
import torch.nn as nn
from src.end_to_end.model.module import Attention, DecoderRNN, EncoderRNN


class E2EChar2CharSpellCheckRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        encoder_emb_dim,
        decoder_emb_dim,
        encoder_hid_dim,
        decoder_hid_dim,
        encoder_dropout=0.1,
        decoder_dropout=0.1,
        attention=True,
        **kwargs
    ) -> None:
        super().__init__()

        self.encoder = EncoderRNN(
            input_dim=input_dim,
            encoder_emb_dim=encoder_emb_dim,
            encoder_hid_dim=encoder_hid_dim,
            encoder_fc_hid_dim=decoder_hid_dim,
            encoder_dropout=encoder_dropout,
            **kwargs
        )
        if not attention:
            attention = None
        else:
            attention = Attention(encoder_hid_dim, decoder_hid_dim, **kwargs)
        self.decoder = DecoderRNN(
            output_dim=output_dim,
            decoder_emb_dim=decoder_emb_dim,
            encoder_hid_dim=encoder_hid_dim,
            decoder_hid_dim=decoder_hid_dim,
            decoder_dropout=decoder_dropout,
            attention=attention,
            **kwargs
        )

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: Batch x Length
        trg: Batch x Length
        outputs: Batch x Length x vocab_size
        """
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
        outputs = torch.zeros(trg.shape[0], trg.shape[1], self.decoder.output_dim)

        encoder_outputs, encoder_hidden = self.encoder(src)

        decoder_hidden = encoder_hidden
        decoder_input = trg[0, :]  # <sos> charter
        for time_step in range(1, trg.shape[0]):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs[time_step] = decoder_output
            top1 = decoder_output.argmax(1)

            # Teacher force
            rnd_teacher = torch.rand(1).item()
            teacher_force = rnd_teacher < teacher_forcing_ratio
            decoder_input = trg[time_step, :] if teacher_force else top1

        outputs = outputs.transpose(0, 1).contiguous()
        return outputs

    def forward_encoder(self, src):
        src = src.permute(1, 0)
        outputs, hidden = self.encoder(src)
        return outputs, hidden

    def forward_decoder(self, trg, decoder_hidden, encoder_outputs):
        """
        trg: TimeStep x Batch
        decoder_hidden: Batch x Decoder_hidden_dim
        encoder_outputs: Length x Bactch x (Encoder_hidden_dim * 2 if bidirectional = True else 1)

        decoder_output: batch_size x 1 x vocab_size
        """
        decoder_input = trg[-1]

        decoder_output, decoder_hidden = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        decoder_output = decoder_output.unsqueeze(1)

        return decoder_output, (decoder_hidden, encoder_outputs)
