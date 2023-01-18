import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

plt.switch_backend('agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=device))


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)  # Could be LogSoftmax

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(hidden[0][0]))
        return output, hidden


class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


class AttnDecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.5):
        super(AttnDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.Ua = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wa = nn.Linear(self.hidden_size, self.hidden_size)
        self.va = nn.Parameter(torch.randn(1, hidden_size), requires_grad=True)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_hiddens):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        encoder_hiddens = encoder_hiddens.unsqueeze(1)
        attn_weights = F.softmax(torch.inner(
            self.va, self.tanh(self.Ua(encoder_hiddens) + self.Wa(hidden))), dim=1)

        context = torch.sum(
            torch.mul(attn_weights, encoder_hiddens.squeeze()), dim=1)

        output = torch.cat((embedded[0], context), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        cat_output = torch.cat((context, hidden[0]), 1)
        output = F.log_softmax(self.out(cat_output), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
