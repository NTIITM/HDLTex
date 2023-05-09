import pathlib

import numpy as np
import torch
import torch.utils.data as data
import tqdm
import dataset
import DataLoader
import Tokenizer
import utils
import random
import lower_model
import ipdb
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')


class trainer():

    def __init__(self, data_path, bert_path=None):
        self.data_path = data_path

    def get_data_and_tokenzier(self, num_accept=1000):
        path_X = self.data_path / "X.txt"
        path_Y = self.data_path / "Y.txt"
        X, Y = DataLoader.get_data_and_labels(path_X, path_Y)
        tokenizer = Tokenizer.tokenizer()
        # done 加载速度很慢，是否可以换成多线程,或者预加载,在这里加上一个判别文件是否存在的逻辑
        before_tokenizer_w2i = pathlib.Path(self.data_path / "w2l.json")
        before_tokenizer_i2w = pathlib.Path(self.data_path / "l2w.json")
        if before_tokenizer_w2i.exists():
            utils.Log.info("找到初始化词表，正在加载")
            assert before_tokenizer_i2w.exists()
            # 用工具类中的方法回比较方便
            tokenizer.word_to_id = utils.OrJson.load(self.data_path / "w2l.json")
            tokenizer.id_to_word = utils.OrJson.load(self.data_path / "l2w.json")
        else:
            utils.Log.info("无初始化此表，正在生成中")
            # for x in tqdm.tqdm(X, desc="初始化词表中"):
            tokenizer.fit_on_texts(X, num_accept=num_accept)
            utils.OrJson.dump(tokenizer.word_to_id, self.data_path / "w2l.json")
            utils.OrJson.dump(tokenizer.id_to_word, self.data_path / "l2w.json")
        # print(tokenizer.word_to_id)
        return X, Y, tokenizer
    def convert_X_to_vec(self,X):
        ret = []
        for x in tqdm.tqdm(X,desc="正将训练数据转为向量形式"):
            tem = []
            for word in x:
                try:
                    tem.append(np.array(wv[word]))
                except KeyError:
                    # print(word)
                    continue
            ret.append(torch.tensor(np.array(tem)))
        return ret
            # 通过给定的数据训练对应的模型,实现一个函数训练分层模型
    def train_model(self, X, Y, num_classes, model_path, model_name, skip=False, Batch_size=32,
                    loss=torch.nn.CrossEntropyLoss(reduction="mean"),
                    net=None,
                    device=torch.device("cuda:0"), num_epoch=300,lr = 1e-3):
        if (model_path / model_name).exists() and skip:
            #  处理已有模型文件时,不需要再次训练
            return
        utils.Log.info(f"开始训练模型")
        # model_path = pathlib.Path().cwd() / model_name
        net = lower_model.lower_model(num_classes=num_classes,dnn_dropout=0,rnn_dropout=0).to(device)
        print(net)
        X = self.convert_X_to_vec(X)
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        Y = torch.tensor(Y)
        # print(Y)
        source = dataset.dataSet(X, Y)
        n_train = len(source)
        split = n_train // 3
        indices = list(range(n_train))
        random.shuffle(indices)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        train_loader = torch.utils.data.DataLoader(
            dataset=source,
            sampler=train_sampler,
            batch_size=Batch_size,

        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=source,
            sampler=valid_sampler,
            batch_size=Batch_size,
        )
        optimizer = torch.optim.Adam(net.parameters(),lr=lr)
        train_acc = 0
        valid_acc = 0
        total = 0
        correct = 0
        for i in range(num_epoch):
            loop = tqdm.tqdm(train_loader)
            net.train()
            for x, y in loop:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = net(x)
                # print(y_hat,y)
                t_l = loss(y_hat, y)
                t_l.backward()
                # ipdb.set_trace()
                optimizer.step()
                # print(list(net.parameters()))
                total += y_hat.shape[0]
                correct += (torch.argmax(y_hat, dim=1) == y).sum()
                train_acc = correct / total
                loop.set_postfix(loss=t_l.item(), acc=train_acc.item())
                loop.set_description(desc=f"Epoch{i + 1}/{num_epoch}")
            # net.eval()
            # loop = tqdm.tqdm(valid_loader)
            # for x, y in loop:
            #     x = x.to(device)
            #     y = y.to(device)
            #     y_hat = net(x)
            #     v_l = loss(y_hat, y)
            #     total += y_hat.shape[0]
            #     correct += (torch.argmax(y_hat, dim=1) == y).sum()
            #     valid_acc = correct / total
            #     loop.set_postfix(loss=v_l.item(), acc=valid_acc.item())
            #     loop.set_description(desc=f"Epoch{i + 1}/{num_epoch},测试中")
            torch.save(net, model_path / model_name)
        info = {
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "model_name": model_name,
        }
        utils.Log.info("将日志信息写入")
        utils.Log.add_info(model_path / "log.json", info)



if __name__ == '__main__':
    paths = ["../data/WOS5736", "../data/WOS11967", "../data/WOS46985"]
    for path in paths:
        train = trainer(data_path=pathlib.Path(path))
        # X, Y1, Y2 = DataLoader.get_data_and_hierarchical_label(pathlib.Path(path))
        X,Y = DataLoader.get_data_and__label(pathlib.Path(path))
        total_len = len(X)
        # _, _, tokenize = train.get_data_and_tokenzier(num_accept=2000)
        low_level_nums = max(Y) + 1
        train.train_model(X, Y, num_classes=low_level_nums, model_path=pathlib.Path(path),
                          model_name='low_level_model', skip=False)
        x_container = [[] for i in range(low_level_nums)]
        y_container = [[] for i in range(low_level_nums)]
        numclasser_container = [0] * low_level_nums
        #
        # 训练一个分出一类标签的数据

    # train.train()
