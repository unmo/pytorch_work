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
            self.constructions[f"{layer}"] = torch.nn.Linear(dims[0], dims[1])
            # exec(f"self.fc{layer} = torch.nn.Linear({dims[0]}, {dims[1]})")

    def __get_conbinate_relation(self):
        prev_dim = self.init_dim
        relations = []
        for layer in range(1, self.layer_num + 1):
            if prev_dim == 1:
                prev_dim *= 2
            relations.append((prev_dim, prev_dim // 2))
            prev_dim //= 2

        return relations

    def forward(self, x):
        for layer in range(1, len(self.conbinate_relations)):
            x = self.constructions[f"{layer}"](x)
            # x = eval(f"torch.relu(self.fc{layer}(x))")
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
            # self.constructions[f"{layer}"] = torch.nn.Linear(dims[0], dims[1])
            exec(f"self.fc{layer} = torch.nn.Linear({dims[1]}, {dims[0]})")

    def forward(self, x):
        for layer in range(1, len(self.conbinate_relations)):
            x = eval(f"torch.relu(self.fc{layer}(x))")
        return x


class StackedAutoEncoder(torch.nn.Module):
    def __init__(self, dim, layer_num):
        super().__init__()
        self.enc = Encoder(dim, layer_num)
        self.dec = Decoder(self.enc.conbinate_relations)

    def forward(self, x):
        self.middle = self.enc(x)
        x = self.dec(self.middle)
        return x


class MyAutoEncoder:
    def __init__(self, dim, layer, lossfunc, optimaizer):
        self.dim = dim
        self.layer = layer
        self.optimaizer = optimaizer
        self.stacked_ae = StackedAutoEncoder(dim, layer)

    def train(self, train_data, epochs):
        losses = []
        output_and_label = []

        for epoch in range(1, epochs + 1):
            pass


if __name__ == "__main__":
    # e = Encoder(256, 5)
    # print(e.conbinate_relations)
    # out = e.forward(torch.rand(2, 256))
    # d = Decoder(e.conbinate_relations)
    # out = d.forward(out)
    # print(out.shape)
    dim = 256
    layer = 5
    autoencder = StackedAutoEncoder(dim, layer)
    out = autoencder.forward(torch.rand(10, dim))
    print(out.shape)
    print(autoencder.state_dict())

    lossfunc = torch.nn.MSELoss()
    optimaizer = torch.optim.SGD(autoencder.parameters(), lr=0.1)

    my_ae = MyAutoEncoder(dim, layer, lossfunc, optimaizer)
