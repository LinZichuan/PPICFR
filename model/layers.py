import math

import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def sigmod2(y):
    # y = torch.clamp(0.995 / (1.0 + torch.exp(-y)) + 0.0025, 0, 1)
    # y = torch.clamp(y, -16, 16)
    y = torch.sigmoid(y)
    # y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025

    return y

def safe_sqrt(x):
    ''' Numerically safe version of Pytoch sqrt '''
    return torch.sqrt(torch.clip(x, 1e-9, 1e+9))

class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, cfg, device):
        super(ShareNetwork, self).__init__()
        if cfg.BatchNorm1d == 'true':
            print("use BatchNorm1d")
            self.DNN = nn.Sequential(

                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, share_dim),
                # nn.BatchNorm1d(share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, base_dim),
                # nn.BatchNorm1d(base_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate)
            )
        else:
            print("No BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
            )

        self.DNN.apply(init_weights)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        h_rep = self.DNN(x)
        if self.cfg.normalization == "divide":
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep_norm

class BaseModel(nn.Module):
    def __init__(self, base_dim, cfg):
        super(BaseModel, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate)
        )
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits

class PrpsyNetwork(nn.Module):
    """propensity network"""

    def __init__(self, base_dim, cfg):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p

class Mu0Network(nn.Module):
    def __init__(self, base_dim, cfg):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p

class Mu1Network(nn.Module):
    def __init__(self, base_dim, cfg):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p

class TauNetwork(nn.Module):
    """pseudo tau network"""

    def __init__(self, base_dim, cfg):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        # return self.tanh(p)
        return tau_logit

class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim, dropout, device):
        super(TransformerNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = self._generate_positional_encoding(hidden_dim, 10000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.device = device
        self.to(device)
        self.positional_encoding = self.positional_encoding.to(device)

    def _generate_positional_encoding(self, hidden_dim, max_len):
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        x = torch.mean(x, dim=1)  # 对时间步维度进行平均池化
        return x
