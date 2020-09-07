# -*-coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


class InputDataFactory:
    """Dataクラス生成クラス
    """

    def __init__(self, data_class: "Data"):
        self.data = data_class

    def get_data(self):
        return self.data.get_data()


class TmplateData(metaclass=ABCMeta):
    @abstractmethod
    def get_data(self):
        pass


class DummyInputData1(TmplateData):
    """ダミーのアルゴリズムIFの入力データクラス
    アルゴリズムの機能テストに用いる想定

    Args:
        Data (Data): Dataクラスオブジェクト
    """

    def __init__(self, data_size: list):
        self.input_data = np.random.random([data_size[0], data_size[1]])

    def get_data(self):
        return self.input_data


class InputData(TmplateData):
    """アルゴリズムIFの入力データクラス

    Args:
        Data (Data): Dataクラスオブジェクト
    """

    def __init__(self):
        """入力データ生成
        """
        self.input_data = 2

    def get_data(self):
        return self.input_data


if __name__ == "__main__":
    dummy_data = InputDataFactory(DummyInputData1([300, 256]))
    input_data = InputDataFactory(InputData())

    data1 = dummy_data.get_data()
    data2 = input_data.get_data()

    print(data1)
    print(data2)
