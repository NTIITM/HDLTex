"""
加载数据
"""
import pathlib


def get_data_and_labels(path_X, path_Y):
    assert path_X is not None and path_Y is not None
    X = []
    Y = []
    with open(path_X, "r") as x:
        X = x.readlines()
    with open(path_Y, "r") as y:
        Y = y.readlines()
    return X, Y


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi(orjson.loads(pathlib.Path().cwd()))
    x, y = get_data_and_labels(pathlib.Path("../data/WOS5736/X.txt"), pathlib.Path('../data/WOS5736/Y.txt'))
    print(x[0])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
