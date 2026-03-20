import torch.nn as nn
import torch.nn.functional as F

# Q网络结构
#  4 层 MLP：256 -> 128 -> 64 -> 32 -> num_actions


class SimpleQ(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(SimpleQ, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_actions)
        self.dropout = nn.Dropout(0.5)
        
        # Use Xavier uniform distribution to initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def Q_construct(input_dim, num_actions):  # 定义一个函数来构建Q网络，输入参数是输入维度和动作数量，返回一个SimpleQ实例
    return SimpleQ(input_dim=input_dim, num_actions=num_actions)

