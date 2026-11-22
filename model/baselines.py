import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge

from model.layers import Mu0Network, Mu1Network, PrpsyNetwork, ShareNetwork


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class MyPipeline:
    def __init__(self, steps=[]):
        if not steps:
            self.pipeline = None
        elif len(steps) == 1:
            self.pipeline = steps[0][1]
        else:
            self.pipeline = Pipeline(steps=steps)

    def fit(self, X_train):
        if self.pipeline is not None:
            self.pipeline.fit(X_train)

    def fit_transform(self, X_train):
        if self.pipeline is not None:
            return self.pipeline.fit_transform(X_train)
        return X_train

    def transform(self, X_train):
        if self.pipeline is not None:
            return self.pipeline.transform(X_train)
        return X_train

def make_base_regressor(model_name, input_dim, device, cfg, binary, epochs):
    if '_linear' in model_name.lower():
        return Linear_model(binary=binary, epochs=epochs)
    if "_mlp" in model_name.lower():
        return MLP(input_dim, cfg.share_dim, device, cfg.lr, cfg.l2, cfg.do_rate, binary, epochs=epochs)

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, lr, weight_decay, do_rate=0.1, binary=False, epochs=100):
        super(MLP, self).__init__()
        self.DNN = nn.Sequential(

            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(hidden_dim, 1),
        )
        self.binary = binary
        if binary:
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Identity()
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epochs = epochs
        self.to(device)

    def forward(self, x):
        output = self.DNN(x)
        output = self.output_layer(output)
        return output

    def fit(self, inputs, labels):

        print("inputs shape:", inputs.shape)
        print("labels shape:", labels.shape)
        inputs = torch.from_numpy(inputs).float().to(self.device)
        labels = torch.from_numpy(labels.reshape((-1, 1))).float().to(self.device)

        datasets = MyDataset(inputs, labels)
        dataloader = DataLoader(datasets, batch_size=128, shuffle=True)

        loss_fn = nn.BCELoss() if self.binary else nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train()

        best_loss = float('inf')
        epochs_without_improvement = 0
        patience = 5

        for epoch in range(self.epochs):
            epoch_loss = 0
            for i, (batch_inputs, batch_labels) in enumerate(dataloader):
                predictions = self.forward(batch_inputs)
                loss = loss_fn(predictions, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                print("Epoch {}, batch {}: Loss = {}".format(epoch, i, loss))

            avg_epoch_loss = epoch_loss / len(dataloader)
            print("Epoch {}: Average Loss = {}".format(epoch, avg_epoch_loss))

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered after {} epochs without improvement.".format(patience))
                break

    def predict(self, inputs):
        self.eval()
        inputs = torch.from_numpy(inputs).float().to(self.device)
        outputs = self.forward(inputs)
        return outputs.cpu().detach().numpy()

class Linear_model():
    def __init__(self, binary=False, epochs=100):
        self.binary = binary
        if binary:
            self.reg = LogisticRegression(max_iter=epochs)
        else:
            # self.reg = LinearRegression()
            self.reg = Ridge(alpha=1.0, max_iter=epochs)
        self.epochs = epochs

    def fit(self, inputs, labels):
        labels = labels.flatten()
        self.reg.fit(inputs, labels)

    def predict(self, inputs):
        if self.binary:
            return self.reg.predict_proba(inputs)[:, 1].reshape((-1, 1))
        return self.reg.predict(inputs).reshape((-1, 1))

class DRCFR(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, cfg, device, next_dim=0):
        super(DRCFR, self).__init__()
        # 三个表示学习网络
        self.gamma_net = ShareNetwork(input_dim, share_dim, base_dim, cfg, device)
        self.delta_net = ShareNetwork(input_dim, share_dim, base_dim, cfg, device)
        self.upsilon_net = ShareNetwork(input_dim, share_dim, base_dim, cfg, device)
        
        # 处理分配网络(使用gamma和delta)
        self.prpsy_network = PrpsyNetwork(2 * base_dim, cfg)
        
        # 结果预测网络(使用delta和upsilon)
        self.mu1_network = Mu1Network(2 * base_dim+next_dim, cfg)
        self.mu0_network = Mu0Network(2 * base_dim+next_dim, cfg)
        
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x, next_share_h=None):
        # 提取三个因子
        gamma = self.gamma_net(x)
        delta = self.delta_net(x)
        upsilon = self.upsilon_net(x)
        
        # 预测处理分配概率
        t_input = torch.cat([gamma, delta], dim=1)
        p_prpsy_logit = self.prpsy_network(t_input)
        p_prpsy = (torch.sigmoid(p_prpsy_logit) + 0.01) / 1.02
        
        # 预测潜在结果
        if next_share_h is not None:
            y_input = torch.cat([delta, upsilon, next_share_h.reshape(delta.shape[0], -1)], dim=1)
        else:
            y_input = torch.cat([delta, upsilon], dim=1)
        mu1_logit = self.mu1_network(y_input)
        mu0_logit = self.mu0_network(y_input)
        
        # p_mu1 = sigmod2(mu1_logit)
        # p_mu0 = sigmod2(mu0_logit)

        p_mu1 = mu1_logit
        p_mu0 = mu0_logit
        
        # 计算tau
        p_tau = p_mu1 - p_mu0
        
        # 计算factual和counterfactual预测
        p_yf = p_mu1 * p_prpsy + p_mu0 * (1 - p_prpsy)
        p_ycf = p_mu0 * p_prpsy + p_mu1 * (1 - p_prpsy)
        
        return p_prpsy_logit, p_prpsy, p_mu1, p_mu0, p_tau, p_yf, p_ycf, gamma, delta, upsilon

class PrpsyNetworkTrainer:
    def __init__(self, base_dim, cfg, lr=0.001, device='auto'):
        """
        Propensity Network Trainer
        Args:
            base_dim: 输入特征维度
            cfg: BaseModel配置参数
            lr: 学习率 (default: 0.001)
            device: 训练设备 ('cuda', 'cpu' 或 'auto')
        """
        # 自动检测设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device == 'auto' else torch.device(device)
        
        # 初始化模型
        self.model = PrpsyNetwork(base_dim, cfg).to(self.device)
        
        # 设置优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()  # 自动包含sigmoid

    def fit(self, x_train, t_train, epochs=10, batch_size=32, verbose=True):
        """
        训练网络
        Args:
            x_train: 训练数据 (numpy array 或 torch.Tensor)
            t_train: 训练标签 (numpy array 或 torch.Tensor)
            epochs: 训练轮次 (default: 10)
            batch_size: 批大小 (default: 32)
            verbose: 是否打印训练信息 (default: True)
        """
        # 转换数据为张量
        x_tensor = torch.from_numpy(x_train).float().to(self.device)
        t_tensor = torch.from_numpy(t_train.reshape((-1, 1))).float().to(self.device)

        # 创建数据加载器
        datasets = MyDataset(x_tensor, t_tensor)
        dataloader = DataLoader(datasets, batch_size=128, shuffle=True)

        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                # 数据迁移到设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 记录损失
                epoch_loss += loss.item() * inputs.size(0)

            # 计算平均损失
            epoch_loss /= len(dataloader.dataset)
            
            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

    def predict(self, x):
        """
        预测概率
        Args:
            x: 输入数据 (numpy array 或 torch.Tensor)
        Returns:
            预测概率 (numpy array)
        """
        self.model.eval()
        with torch.no_grad():
            # 转换输入数据
            x_tensor = torch.as_tensor(x, dtype=torch.float32).to(self.device)
            
            # 预测并应用sigmoid
            logits = self.model(x_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return probabilities.squeeze()
