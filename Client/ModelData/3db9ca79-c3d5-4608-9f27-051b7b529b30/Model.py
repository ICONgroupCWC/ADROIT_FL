import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, time_steps=None, hidden_size=64):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out[:, -1, :])  # Take the last time step output

        # Dense layers
        fc1_out = self.relu(self.fc1(lstm2_out))
        output = self.fc2(fc1_out)

        return output
