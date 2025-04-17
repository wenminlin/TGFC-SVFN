# 修改后的噪声预测网络
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
# from src.CrossmodalTransformer import MULTModel


    





class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb







# 这段代码的主要作用是对输入文本特征 x 进行处理，去除噪声
class Text_Noise_Pre(nn.Module):
    def __init__(self, T, ch,dropout, in_ch):
        super().__init__()
        # assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = int(in_ch / 2)
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.fc1 = nn.LSTM(input_size=in_ch, hidden_size=int(in_ch/2), num_layers=2)
        self.fc1_1 = nn.LSTM(input_size=in_ch, hidden_size=int(in_ch / 2), num_layers=2)
        self.fc2 = nn.LSTM(input_size=int(in_ch / 2), hidden_size=in_ch, num_layers=2)

        self.dropout = dropout
        self.swish = Swish()

    def forward(self, x, t, y):
        # Timestep embedding
        temb = self.time_embedding(t)[:, None, :]
        h, _ = self.fc1(x)
        h_y, _ = self.fc1_1(y)

        h = h + h_y + temb
        h = self.swish(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h, _ = self.fc2(h)
        return h




class Audio_Noise_Pre(nn.Module):
    def __init__(self, T, ch,dropout, in_ch):
        super().__init__()
        # assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = int(in_ch/2)
        self.time_embedding = TimeEmbedding(T, in_ch, tdim)

        self.fc1 = nn.Linear(in_ch, int(in_ch/2))
        self.fc1_1 = nn.Linear(in_ch, int(in_ch / 2))
        self.fc2 = nn.Linear(int(in_ch/2), in_ch)
        self.dropout = dropout
        self.swish = Swish()

    def forward(self, x, t, y):
        # Timestep embedding
        temb = self.time_embedding(t)[:, None, :]
        h = self.fc1(x)
        h_y = self.fc1_1(y)
        h = h + temb + h_y
        h = self.swish(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)
        return h

    
    
class Visual_Noise_Pre(nn.Module):
    def __init__(self, T, ch, dropout, in_ch):
        super().__init__()
        # assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = int(in_ch/2)
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.fc1 = nn.Linear(in_ch, int(in_ch / 2))
        self.fc1_1 = nn.Linear(in_ch, int(in_ch / 2))
        self.fc2 = nn.Linear(int(in_ch / 2), in_ch)
        self.dropout = dropout
        self.swish = Swish()

    def forward(self, x, t, y):
        # Timestep embedding
        temb = self.time_embedding(t)[:, None, :]
        h = self.fc1(x)
        h_y = self.fc1_1(y)
        h = h + temb + h_y
        h = self.swish(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)
        return h

if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)
