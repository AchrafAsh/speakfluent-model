from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import transforms

import numpy as np

from preprocessing import SpeakDataset, pad_collate, TextTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MVPModel(nn.Module):
    def __init__(self, batch_size: int):
        super(MVPModel, self).__init__()
        self.rnn = nn.LSTM(input_size=128+28, hidden_size=20,
                           num_layers=2, bidirectional=True)
        self.fc = nn.Linear(40, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, specgram, label, initial_states):
        # initial states
        h0, c0 = initial_states

        x = torch.cat((label, specgram), 2)

        output, (hn, cn) = self.rnn(x, (h0, c0))
        output = self.fc(output)
        return self.softmax(output)


def train(model, loss_fn,
          train_loader, valid_loader,
          epochs, optimizer, train_losses,
          valid_losses, change_lr=None):

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        batch_losses = []

    if change_lr:
        optimizer = change_lr(optimizer, epoch)

    for i, sample in enumerate(train_loader):
        specgram, label, score = sample
        specgram = specgram[:, :, 0, :].to(device)
        label = label.to(device)
        score = score.to(device)
        _, word_size, _ = label.size()

        # initial states
        h0 = torch.ones(4, word_size, 20)
        c0 = torch.ones(4, word_size, 20)

        score_pred = model(specgram, label, (h0, c0))[:, :, 0]

        loss = loss_fn(score_pred, score)
        loss.backward()
        batch_losses.append(loss.item())
        optimizer.step()

    train_losses.append(batch_losses)
    print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
    model.eval()
    batch_losses = []
    trace_score = []
    trace_score_pred = []

    for i, data in enumerate(valid_loader):
        specgram, label, score = sample
        specgram = specgram[:, :, 0, :].to(device)
        label = label.to(device)
        score = score.to(device)
        _, word_size, _ = label.size()

        # initial states
        h0 = torch.ones(4, word_size, 20)
        c0 = torch.ones(4, word_size, 20)

        score_pred = model(specgram, label, (h0, c0))[:, :, 0]

        loss = loss_fn(score_pred, score)
        trace_score.append(score.cpu().detach().numpy())
        trace_score_pred.append(score_pred.cpu().detach().numpy())
        batch_losses.append(loss.item())

    valid_losses.append(batch_losses)
    trace_score = np.concatenate(trace_score)
    trace_score_pred = np.concatenate(trace_score_pred)
    accuracy = np.mean(trace_score_pred.argmax(axis=1) == trace_score)
    print(
        f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')


if __name__ == '__main__':
    # load data
    batch_size = 4
    dataset = SpeakDataset(csv_file='data.csv', root_dir='data')
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=pad_collate)
    valid_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=pad_collate)
    text_transformer = TextTransform()
    model = MVPModel(batch_size=batch_size)
    learning_rate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 50
    loss_fn = nn.MSELoss()
    train_losses = []
    valid_losses = []
    train(model, loss_fn,
          train_loader, valid_loader,
          epochs, optimizer,
          train_losses, valid_losses)
torch.save(model.state_dict(), "speakfluent_mvp.pth")
