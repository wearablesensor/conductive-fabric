import torch
import torch.nn as nn
import torch.nn.functional as F
class GlobalAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GlobalAttention, self).__init__()
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: input tensor of shape [batch_size, sequence_length, input_size]
        q = self.Wq(x)  # [batch_size, sequence_length, hidden_size]
        k = self.Wk(x)  # [batch_size, sequence_length, hidden_size]
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, sequence_length, sequence_length]
        # Apply scaling
        scores = scores / math.sqrt(self.hidden_size)
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)
        # Compute weighted sum of values
        attended_values = torch.bmm(attention_weights.transpose(1, 2), x)  # [batch_size, hidden_size, 1]

        return attended_values.squeeze(2)
class LocalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(LocalAttention, self).__init__()
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)
        self.kernel_size = kernel_size


    def forward(self, x):
        # x: input tensor of shape [batch_size, sequence_length, input_size]
        q = self.Wq(x)  # [batch_size, sequence_length, hidden_size]
        k = self.Wk(x)  # [batch_size, sequence_length, hidden_size]
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, sequence_length, sequence_length]

        # Apply masking to create local attention
        mask = torch.zeros(scores.size(), dtype=torch.bool, device=x.device)
        for i in range(x.size(1)):
            start = max(0, i - self.kernel_size + 1)
            end = min(x.size(1), i + self.kernel_size)
            mask[:, i, start:end] = True
        scores[~mask] = float('-inf')
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)
        # Compute weighted sum of values
        attended_values = torch.bmm(attention_weights, x) # [batch_size, sequence_length, input_size]

        return attended_values

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size *2, num_classes)

    def forward(self, x):
        # x: input tensor of shape [batch_size, sequence_length, input_size]
        out, _ = self.lstm(x)  # [batch_size, sequence_length, hidden_size*2]
        out = self.fc(out[:, -1, :])  # [batch_size, num_classes]
        return out


class cnn_Lstm_att(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, kernel_size):
        super(cnn_Lstm_att, self).__init__()

        self.global_attention = GlobalAttention(input_size, hidden_size)
        self.local_attention = LocalAttention(input_size, hidden_size, kernel_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.bilstm = BiLSTM(4, hidden_size, num_layers, num_classes)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        # x: input tensor of shape [batch_size, sequence_length, input_size]
        global_out = self.global_attention(x)  # [batch_size, hidden_size]

        local_out = self.local_attention(x)  # [batch_size, input_size]

        out = torch.cat((global_out, local_out), dim=1)  # [batch_size, hidden_size*2]
        x= self.cnn(out)
        out_log = self.bilstm(x)  # [batch_size, num_classes]
        output = self.out(out_log)
        return out_log, output
