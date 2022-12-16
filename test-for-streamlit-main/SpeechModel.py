import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.ConvLayer=nn.Sequential(
        nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride),
        nn.BatchNorm1d(n_channel),
        nn.ReLU(),
        nn.MaxPool1d(4),
        nn.Conv1d(n_channel, n_channel, kernel_size=3),
        nn.BatchNorm1d(n_channel),
        nn.ReLU(),
        nn.MaxPool1d(4),
        nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
        nn.BatchNorm1d(2 * n_channel),
        nn.ReLU(),
        nn.MaxPool1d(4),
        nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3),
        nn.BatchNorm1d(2 * n_channel),
        nn.ReLU(),
        nn.MaxPool1d(4)
        )
        self.fc=nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.ConvLayer(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)

def tensor2label(x):
    label=['backward',
     'bed',
     'bird',
     'cat',
     'dog',
     'down',
     'eight',
     'five',
     'follow',
     'forward',
     'four',
     'go',
     'happy',
     'house',
     'learn',
     'left',
     'marvin',
     'nine',
     'no',
     'off',
     'on',
     'one',
     'right',
     'seven',
     'sheila',
     'six',
     'stop',
     'three',
     'tree',
     'two',
     'up',
     'visual',
     'wow',
     'yes',
     'zero']
    return label[torch.argmax(x,dim=-1)]