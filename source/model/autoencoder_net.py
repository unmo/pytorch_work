# -*-coding: utf-8 -*-

import torch

# ネットワーク定義を動的に行うのはネットワークモデルの評価時のみにしか使わない
# ネットワーク構造が確定したら、内部構造をコーディングすればよい


class Encoder(torch.nn.Module):
    def __init__(self, dim: int, layer_num: int):
        super().__init__()
        self.init_dim = dim
        self.layer_num = layer_num
        self.constructions = {}
        self.conbinate_relations = self.__get_conbinate_relation()

        for layer, dims in enumerate(self.conbinate_relations, 1):
            # self.constructions[f"{layer}"] = torch.nn.Linear(dims[0], dims[1])
            exec(f"self.fc{layer} = torch.nn.Linear({dims[0]}, {dims[1]})")

    def __get_conbinate_relation(self):
        prev_dim = self.init_dim
        relations = []
        for layer in range(1, self.layer_num + 1):
            if prev_dim == 1:
                relations.append((prev_dim, prev_dim))
            else:
                relations.append((prev_dim, prev_dim // 2))
                prev_dim //= 2

        return relations

    def forward(self, x):
        for layer in range(1, len(self.conbinate_relations)):
            # x = self.constructions[f"{layer}"](x)
            x = eval(f"torch.relu(self.fc{layer}(x))")
        return x


class Decoder(torch.nn.Module):
    def __init__(self, conbinate_relations: list):
        """恒等写像なので、エンコーダ側の全結合層の関係を引数に渡す

        Args:
            conbination_relations (list): エンコード側の全結合層の関係
        """
        super().__init__()
        self.constructions = {}
        self.conbinate_relations = conbinate_relations[::-1]

        for layer, dims in enumerate(self.conbinate_relations):
            # self.constructions[f"{layer}"] = torch.nn.Linear(dims[1], dims[0])
            exec(f"self.fc{layer} = torch.nn.Linear({dims[1]}, {dims[0]})")

    def forward(self, x):
        for layer in range(1, len(self.conbinate_relations)):
            # x = self.constructions[f"{layer}"](x)
            x = eval(f"torch.relu(self.fc{layer}(x))")
        return x


class StackedAutoEncoder(torch.nn.Module):
    def __init__(self, dim, layer_num):
        super().__init__()
        self.enc = Encoder(dim, layer_num)
        self.dec = Decoder(self.enc.conbinate_relations)

    def __call__(self, x):
        self.middle = self.enc(x)
        x = self.dec(self.middle)
        return x


class MyAutoEncoder:
    def __init__(self, dim, layer):
        self.dim = dim
        self.layer = layer
        self.stacked_ae = StackedAutoEncoder(dim, layer)
        print(self.stacked_ae.parameters())

    def train(self, train_data, epochs, lossfun, optimizer):
        losses = []
        output_and_label = []

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for x in train_data:
                print(type(x))
                out = self.stacked_ae(x)
                loss = lossfunc(out, x)
                optimaizer.zero_grad()
                loss.backward()
                optimaizer.step()
                print(loss)
                # losses.append(loss.data[0])

            # print(f"epoch [{epoch}/{epochs}, loss: {loss.data[0]}]")


if __name__ == "__main__":

    # モデル定義
    # ファクトリパターンでクラス化するとよい
    dim = 16
    layer = 5
    # autoencder = StackedAutoEncoder(dim, layer)

    my_ae = MyAutoEncoder(dim, layer)
    # train条件
    # trainの条件はファクトリパターンでクラス化する
    epochs = 10
    lossfunc = torch.nn.MSELoss()
    optimaizer = torch.optim.Adam(my_ae.stacked_ae.parameters(), lr=0.01)

    # 自前データセットを生成する場合は、transformでtensor型に変換する
    # 自前データセットを生成する場合は、torch.utils.data.Datasetで定義する
    # train_data = [torch.rand(100, dim) for i in range(100)]
    train_data = [torch.empty(1, dim) for i in range(100)]
    my_ae.train(train_data, epochs, lossfunc, optimaizer)
