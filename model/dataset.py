from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class dataSet(Dataset):

    def __init__(self, data ,label):
        self.data = data
        self.lable = label
        self.length = len(data)

    def __getitem__(self, indexs) -> T_co:
        return self.data[indexs], self.lable[indexs]

    def __len__(self):
        return self.length
