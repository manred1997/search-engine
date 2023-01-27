import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate=0.0) -> None:
        super().__init__()

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class Attention(nn.Module):
    def __init__(
        self, encoder_hid_dim, decoder_hid_dim, attention_type="bahdanau", **kwargs
    ):
        super().__init__()

        self.attention_type = attention_type

        encoder_bidirectional = kwargs.get("encoder_bidirectional", False)
        if encoder_bidirectional:
            input_attention_dim = encoder_hid_dim * 2 + decoder_hid_dim
        else:
            input_attention_dim = encoder_hid_dim * 1 + decoder_hid_dim

        # Alignment score
        self.attn = nn.Linear(input_attention_dim, decoder_hid_dim)

        self.v = nn.Linear(decoder_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: Batch x Decoder_hidden_dim
        encoder_outputs: Length x Bactch x (Encoder_hidden_dim * 2 if bidirectional = True else 1)

        outputs: Batch x Length
        """

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(
            1, encoder_outputs.shape[0], 1
        )  # Batch x Length x Decoder_hidden_dim

        encoder_outputs = encoder_outputs.permute(
            1, 0, 2
        )  # Batch x Length x (Encoder_hidden_dim * 2 if bidirectional = True else 1)

        if self.attention_type == "bahdanau":
            alignment_score = torch.cat((decoder_hidden, encoder_outputs), dim=2)
            energy = torch.tanh(
                self.attn(alignment_score)
            )  # Batch x Length x Decoder_hidden_dim
            attention = self.v(energy).squeeze(2)  # Batch x Length
        elif self.attention_type == "luong":
            pass  # TODO
        else:
            raise Exception("For type attention, Only bahdanau, luong is available")
        return F.softmax(attention, dim=1)


class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_emb_dim,
        encoder_hid_dim,
        encoder_fc_hid_dim,
        encoder_dropout,
        **kwargs,
    ) -> None:
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(input_dim, encoder_emb_dim)
        # Config RNN
        num_layers = kwargs.get("encoder_num_layers", 1)
        bidirectional = kwargs.get("encoder_bidirectional", False)
        self.rnn = nn.GRU(
            encoder_emb_dim,
            encoder_hid_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )  # Input L x B x emb_dim -> Output L x B x rnn_hid_dim * 2 if bidirectional=True else 1

        # Fully Connection
        if bidirectional:
            self.num_hidden = 2 * num_layers
        else:
            self.num_hidden = 1 * num_layers

        self.fc = nn.Linear(encoder_hid_dim * self.num_hidden, encoder_fc_hid_dim)
        self.dropout = nn.Dropout(encoder_dropout)

    def forward(self, x):
        """
        x: L x B

        returns:

            outputs: L x B x (Encoder_Hidden_RNN * 2 if bidirectional = True else 1)

            hidden: B X Decoder_Hidden_RNN

        """
        # Forward embedding -> L X B X emb_dim
        embedded = self.embedding(x)

        # Forward Dropout -> L X B X emb_dim
        embedded = self.dropout(embedded)

        # Forward RNN -> (outputs: L x B x rnn_hid_dim * 2 if bidirectional=True else 1) and (hidden: (num_layer * 2 if bidirectional = True else 1) x B x Hidden_RNN)
        outputs, hidden = self.rnn(embedded)

        # Forward Fully Connected
        place_final_hidden_state = []
        for i in range(self.num_hidden):
            place_final_hidden_state.append(hidden[i, :, :])
        hidden = torch.cat(place_final_hidden_state, dim=-1)
        hidden = self.fc(hidden)
        hidden = torch.tanh(hidden)

        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(
        self,
        output_dim,
        decoder_emb_dim,
        encoder_hid_dim,
        decoder_hid_dim,
        decoder_dropout,
        attention=None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim

        # Embedding
        self.embedding = nn.Embedding(output_dim, decoder_emb_dim)
        # Attention
        self.attention = attention

        encoder_bidirectional = kwargs.get("encoder_bidirectional", False)

        if not self.attention:
            input_dim_rnn = decoder_emb_dim
            input_dim_fc = decoder_hid_dim
        else:

            input_dim_rnn = decoder_emb_dim + (
                encoder_hid_dim * 2 if encoder_bidirectional else 1
            )
            input_dim_fc = (
                decoder_hid_dim
                + decoder_emb_dim
                + (encoder_hid_dim * 2 if encoder_bidirectional else 1)
            )
        # Config RNN
        self.rnn = nn.GRU(input_dim_rnn, decoder_hid_dim)

        # Fully Connection
        self.fc = nn.Linear(input_dim_fc, output_dim)
        self.dropout = nn.Dropout(decoder_dropout)

    def forward(self, input, decoder_hidden, encoder_outputs):
        """
        inputs: Batch
        decoder_hidden: Batch x Decoder_hidden_dim
        encoder_outputs: Length x Batch x (Encoder_hidden_dim * 2 if bidirectional = True else 1)
        """

        input = input.unsqueeze(0)  # 1 x Batch

        embedded = self.embedding(input)  # 1 x Batch x Decoder_hidden_dim
        embedded = self.dropout(embedded)  # 1 x Batch x Decoder_hidden_dim

        # Attention
        if self.attention:
            attention_score = self.attention(decoder_hidden, encoder_outputs)

            attention_score = attention_score.unsqueeze(1)

            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            weighted = torch.bmm(attention_score, encoder_outputs)

            weighted = weighted.permute(1, 0, 2)

            rnn_input = torch.cat((embedded, weighted), dim=2)

        else:
            rnn_input = embedded

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        if self.attention:
            embedded = embedded.squeeze(0)
            output = output.squeeze(0)
            weighted = weighted.squeeze(0)

            prediction = self.fc(torch.cat((output, weighted, embedded), dim=1))

        else:
            output = output.squeeze(0)
            prediction = self.fc(output)
        return prediction, decoder_hidden.squeeze(0)
