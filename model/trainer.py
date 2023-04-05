import pathlib
import torch
import torch.utils.data as data
import tqdm
import dataset
import DataLoader
import Tokenizer
import utils
import random
import lower_model


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

    # 通过给定的数据训练对应的模型,实现一个函数训练分层模型
    def train_model(self, X, Y, num_classes, tokenize, model_path, model_name, Batch_size=32,
                    loss=torch.nn.CrossEntropyLoss(reduction="mean"),
                    net=None,
                    device=torch.device("cuda:0"), num_epoch=300):
        utils.Log.info(f"开始训练模型")
        # model_path = pathlib.Path().cwd() / model_name
        net = lower_model.lower_model(embedding_size=len(tokenize.word_to_id), num_classes=num_classes).to(device)
        X = [torch.tensor(tokenize.convert_word_to_id(x)) for x in tqdm.tqdm(X, desc="正在将X转化为id形式")]
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        Y = torch.tensor(Y)
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
        optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.1)
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
                optimizer.step()
                total += y_hat.shape[0]
                correct += (torch.argmax(y_hat, dim=1) == y).sum()
                train_acc = correct / total
                loop.set_postfix(loss=t_l.item(), acc=train_acc.item())
                loop.set_description(desc=f"Epoch{i + 1}/{num_epoch}")
            net.eval()
            loop = tqdm.tqdm(valid_loader)
            for x, y in loop:
                x = x.to(device)
                y = y.to(device)
                y_hat = net(x)
                v_l = loss(y_hat, y)
                total += y_hat.shape[0]
                correct += (torch.argmax(y_hat, dim=1) == y).sum()
                valid_acc = correct / total
                loop.set_postfix(loss=v_l.item(), acc=valid_acc.item())
                loop.set_description(desc=f"Epoch{i + 1}/{num_epoch},测试中")
            torch.save(net, model_path / model_name)
        info = {
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "model_name": model_name,
        }
        utils.Log.info("将日志信息写入")
        utils.Log.add_info(model_path / "log.json", info)

    # def train(self, Batch_size=32, loss=torch.nn.CrossEntropyLoss(reduction="mean"), net=None,
    #           device=torch.device("cuda:0"), num_epoch=300, lr=1e-3):
    #     X, Y, tokenize = self.get_data_and_tokenzier(num_accept=5000)
    #     # ipdb.set_trace()
    #     utils.Log.info(f"词表大小为{len(tokenize.word_to_id)}")
    #     model_path = pathlib.Path().cwd() / "lower_model.bin"
    #     if model_path.exists():
    #         utils.Log.info("找到模型并加载")
    #         net = torch.load(model_path)
    #     else:
    #         net = lower_model.lower_model(embedding_size=len(tokenize.word_to_id), num_classes=11).to(device)
    #     X = [torch.tensor(tokenize.convert_word_to_id(x)) for x in tqdm.tqdm(X, desc="正在将X转化为id形式")]
    #     X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    #     # print(X.shape)
    #     Y = torch.tensor(Y)
    #     source = dataset.dataSet(X, Y)
    #     n_train = len(source)
    #     split = n_train // 3
    #     indices = list(range(n_train))
    #     random.shuffle(indices)
    #     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    #     valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    #     train_loader = torch.utils.data.DataLoader(
    #         dataset=source,
    #         sampler=train_sampler,
    #         batch_size=Batch_size,
    #
    #     )
    #     valid_loader = torch.utils.data.DataLoader(
    #         dataset=source,
    #         sampler=valid_sampler,
    #         batch_size=Batch_size,
    #     )
    #     optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.1)
    #     # net.to(device)
    #     for i in range(num_epoch):
    #         loop = tqdm.tqdm(train_loader)
    #         net.train()
    #         total = 0
    #         correct = 0
    #         for x, y in loop:
    #             x = x.to(device)
    #             y = y.to(device)
    #             # print(x.device, y.device)
    #             # print(f"{x.shape}   {y.shape}")
    #             optimizer.zero_grad()
    #             y_hat = net(x)
    #             # print(y_hat, y)
    #             # y_c = torch.nn.functional.one_hot(y, num_classes=11)
    #             t_l = loss(y_hat, y)
    #             t_l.backward()
    #             optimizer.step()
    #             total += y_hat.shape[0]
    #             correct += (torch.argmax(y_hat, dim=1) == y).sum()
    #             acc = correct / total
    #             # accuracy += (y_hat == y).sum()
    #             loop.set_postfix(loss=t_l.item(), acc=acc.item())
    #             loop.set_description(desc=f"Epoch{i + 1}/{num_epoch}")
    #         net.eval()
    #         loop = tqdm.tqdm(valid_loader)
    #         total = 0
    #         correct = 0
    #         for x, y in loop:
    #             x = x.to(device)
    #             y = y.to(device)
    #             y_hat = net(x)
    #             v_l = loss(y_hat, y)
    #             total += y_hat.shape[0]
    #             correct += (torch.argmax(y_hat, dim=1) == y).sum()
    #             acc = correct / total
    #             # accuracy += (y_hat == y).sum()
    #             loop.set_postfix(loss=v_l.item(), acc=acc.item())
    #             loop.set_description(desc=f"Epoch{i + 1}/{num_epoch},测试中")
    #         torch.save(net, 'lower_model.bin')


if __name__ == '__main__':
    paths = ["../data/WOS5736", "../data/WOS11967", "../data/WOS46985"]
    for path in paths:
        train = trainer(data_path=pathlib.Path(path))
        X, Y1, Y2 = DataLoader.get_data_and_hierarchical_label(pathlib.Path(path))
        _, _, tokenize = train.get_data_and_tokenzier(num_accept=2000)
        low_level_nums = max(Y1) +1
        train.train_model(X, Y1, num_classes=low_level_nums, tokenize=tokenize, model_path=pathlib.Path(path),
                          model_name= 'low_level_model')

        # 训练一个分出一类标签的数据

    # train.train()
