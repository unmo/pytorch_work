# -*-coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


class InputDataFactory:
    """Dataクラス生成クラス
    """

    def __init__(self, data_class: "TemplateData"):
        """コンストラクタ

        Args:
            data_class (TemplateData): 抽象化されたDataクラスを継承しているクラス
        """
        self.data = data_class

    def get_data(self):
        return self.data.get_data()


class TmplateData(metaclass=ABCMeta):
    @abstractmethod
    def get_data(self):
        """データ取得のテンプレートメソッド
        """
        pass


class InputData(TmplateData):
    """アルゴリズムIFの入力データクラス
    音声を対象にするなら{Wave, mp3, ...}InputDataクラスになる
    画像を対象にするなら{Jpeg, Png, ...}InputDataクラスになる

    Args:
        Data (Data): Dataクラスオブジェクト
    """

    def __init__(self):
        """入力データ生成
        どうやってデータ生成をするか
        多分ファイルロードとかになると想像
        """
        self.input_data = 2

    def get_data(self):
        return self.input_data


class DummyInputData1(TmplateData):
    """ダミーのアルゴリズムIFの入力データクラス
    任意の2次元配列を生成、取得可能

    アルゴリズムの機能テストに用いる想定

    Args:
        Data (Data): Dataクラスオブジェクト
    """

    def __init__(self, data_size: list):
        """コンストラクタ

        Args:
            data_size (list): 任意の二次元配列のサイズ
        """
        self.input_data = np.random.random([data_size[0], data_size[1]])

    def get_data(self):
        return self.input_data


class MNISTInputData(TmplateData):
    """MNISTデータ
    アルゴリズムの機能テストに用いる想定
    """

    def __init__(self, batch=128, intensity=1.0):
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        self.input_data = DataLoader(
            datasets.MNIST(
                "./data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                ),
            ),
            batch_size=batch,
            shuffle=True,
        )

        test_loader = DataLoader(
            datasets.MNIST(
                "./data",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                ),
            ),
            batch_size=batch,
            shuffle=True,
        )

    def get_data(self):
        return self.input_data


if __name__ == "__main__":
    dummy_data = InputDataFactory(DummyInputData1([300, 256]))
    input_data = InputDataFactory(InputData())

    data1 = dummy_data.get_data()
    data2 = input_data.get_data()

    print(data1)
    print(data2)
