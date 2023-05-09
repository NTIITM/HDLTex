"""
加载数据
"""
import pathlib
import torch.utils.data.dataset as dataset


def get_data_and_labels(path_X, path_Y):
    assert path_X is not None and path_Y is not None
    X = []
    Y = []
    with open(path_X, "r") as x:
        X = x.readlines()
    with open(path_Y, "r") as y:
        Y = y.readlines()
        Y = [int(y.strip('\n')) for y in Y]
    return X, Y


def get_data_and_hierarchical_label(path):
    assert path is not None
    X = []
    Y1 = []
    Y2 = []
    with open(path / "X.txt", "r") as x:
        X = x.readlines()
    with open(path / "YL1.txt", "r") as y:
        Y1 = y.readlines()
        Y1 = [int(y.strip('\n')) for y in Y1]
    with open(path / "YL2.txt", "r") as y:
        Y2 = y.readlines()
        Y2 = [int(y.strip('\n')) for y in Y2]
    return X, Y1, Y2
def get_data_and__label(path):
    assert path is not None
    X = []
    Y = []
    with open(path / "X.txt", "r") as x:
        X = x.readlines()
    with open(path / "Y.txt", "r") as y:
        Y1 = y.readlines()
        Y1 = [int(y.strip('\n')) for y in Y1]
    return X, Y1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi(orjson.loads(pathlib.Path().cwd()))
    x, y = get_data_and_labels(pathlib.Path("../data/WOS5736/X.txt"), pathlib.Path('../data/WOS5736/Y.txt'))
    print(x[0])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
