import torch
import tqdm
import em_model
import gensim.downloader as api

"""
用来训练低层次的模型,做大类区分
"""


class lower_model(torch.nn.Module):
    def __init__(self, num_classes, rnn_kind="", input_size=300, hidden_size=512, rnn_layers=2, embedding_size=50000,
                 rnn_dropout=0.25, dnn_dropout=0.5):
        # self.device = device
        super().__init__()
        if rnn_kind == "gru":
            self.rnn = torch.nn.GRU(input_size, hidden_size, rnn_layers, batch_first=True, dropout=rnn_dropout)
        else:
            self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers,
                                     batch_first=True,
                                     dropout=rnn_dropout)
        self.dense1 = torch.nn.Linear(512, 256)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dnn_dropout)
        self.relu2 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(256, 128)
        self.dropout2 = torch.nn.Dropout(dnn_dropout)
        self.dense3 = torch.nn.Linear(128, num_classes)
        return

    def forward(self, X):
        # print(X.shape)
        # 可能需要在这里修改
        # print(X.shape)
        output, (h_n,c_n) = self.rnn(X)
        X = h_n.permute(1, 0, 2)[:, -1, :]
        X = torch.squeeze(X)
        # print(h_n.shape)
        # print(X.shape)
        X = self.dense1(X)
        X = self.dropout1(X)
        X = self.relu1(X)
        X = self.dense2(X)
        X = self.dropout2(X)
        X = self.relu2(X)
        X = self.dense3(X)
        return X
