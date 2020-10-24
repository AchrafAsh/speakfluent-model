import os
import string
import pandas as pd

import torchaudio

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class TextTransform:
    """ Maps characters to integers and vice versa """

    def __init__(self):
        char_map_str = """
            ' 0
            <SPACE> 1
            a 2
            b 3
            c 4
            d 5
            e 6
            f 7
            g 8
            h 9
            i 10
            j 11
            k 12
            l 13
            m 14
            n 15
            o 16
            p 17
            q 18
            r 19
            s 20
            t 21
            u 22
            v 23
            w 24
            x 25
            y 26
            z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')

    def tensor_to_text(self, tensor):
        word = tensor.transpose(1, 0)
        output = ['']*len(tensor)
        for i in range(len(tensor)):
            for j in range(28):
                if word[j][i] == 1:
                    output[i] = self.index_map[j]
        print(output)
        return ''.join(output)

    def one_hot_enc(self, word):
        """ Returns a sequence of ones and zeros, result of one hot encoding """
        word = self.text_to_int(word)
        word = Variable(torch.tensor(word))
        word = torch.nn.functional.one_hot(word, len(self.index_map))
        return word.transpose(0, 1)


class SpeakDataset(Dataset):
    """ Pronunced words dataset """

    def __init__(self, csv_file: str, root_dir: str, audio_transform=None, text_transform=None):
        """ Args:
            csv_file (string): path to the csv file
            root_dir (string): directory with all the audio files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.audio_transform = audio_transform
        self.text_transform = TextTransform()
        self.data_labels = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(
            self.root_dir, self.data_labels['filename'][idx])
        waveform, sample_rate = torchaudio.load(audio_name)
        word = self.data_labels["word"][idx]
        score = self.data_labels["score"][idx]

        # one hot encoding word
        label = self.text_transform.one_hot_enc(word).transpose(1, 0)

        # spectrogram
        specgram = torchaudio.transforms.MelSpectrogram()(waveform)
        specgram = F.interpolate(specgram, size=len(
            word), mode="nearest").transpose(1, 2)

        # convert to tensor
        score = score.split(';')
        score = list(map(int, score))
        score = torch.FloatTensor(score)

        if self.audio_transform:
            specgram = self.audio_transform(specgram)

        sample = {"specgram": specgram, "label": label, "score": score}

        return specgram, label, score


def pad_collate(batch):
    specgrams = [item[0].transpose(0, 1) for item in batch]
    labels = [item[1] for item in batch]
    scores = [item[2] for item in batch]

    specgrams = pad_sequence(specgrams, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    scores = pad_sequence(scores, batch_first=True, padding_value=0)

    return specgrams, labels, scores
