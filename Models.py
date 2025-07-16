import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchsummary import summary
import math


class GRUModel(nn.Module):
    def __init__(self, input_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size=32, bias=False, batch_first=True, dropout=0.2)
        self.linear1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        output, h_n = self.gru(x)
        # Use only the last hidden state
        last_hidden_state = h_n[-1]
        x = self.linear1(last_hidden_state)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class bidirectionalGRUModel(nn.Module):
    def __init__(self, input_size):
        super(bidirectionalGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size=32, bias=False, batch_first=True, dropout=0.2, bidirectional=True)
        self.linear1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        output, h_n = self.gru(x)
        # Use only the last hidden state
        last_hidden_state = h_n[-1]
        x = self.linear1(last_hidden_state)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GRU2Model(nn.Module):
    def __init__(self, input_size):
        super(GRU2Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size=32, bias=False, batch_first=True, dropout=0.2, num_layers=2)
        self.linear1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        output, h_n = self.gru(x)
        # Use only the last hidden state
        last_hidden_state = h_n[-1]
        x = self.linear1(last_hidden_state)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=32, bias=False, batch_first=True, dropout=0.2)
        self.linear1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        output, h_n = self.lstm(x)
        # Use only the last hidden state
        last_hidden_state = h_n[-1]
        x = self.linear1(last_hidden_state)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class LSTM2Model(nn.Module):
    def __init__(self, input_size):
        super(LSTM2Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=32, bias=False, batch_first=True, dropout=0.2, num_layers=2)
        self.linear1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        output, h_n = self.lstm(x)
        # Use only the last hidden state
        last_hidden_state = h_n[-1]
        x = self.linear1(last_hidden_state)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GRU3Model(nn.Module):
    def __init__(self, input_size):
        super(GRU3Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size=32, bias=False, batch_first=True, dropout=0.2, num_layers=3)
        self.linear1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        output, h_n = self.gru(x)
        # Use only the last hidden state
        last_hidden_state = h_n[-1]
        x = self.linear1(last_hidden_state)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x[-1])
        k = self.key(x)
        v = x
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values


class attentionGRUModel(nn.Module):
    # GRU + Self-Attention Mechanism
    def __init__(self, input_size):
        super(attentionGRUModel, self).__init__()
        self.hidden_size = 32
        self.attention = SelfAttention(self.hidden_size)
        self.gru = nn.GRU(input_size, hidden_size=32, bias=False, batch_first=True, dropout=0.2)
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Use only the last hidden state
        output, h_n = self.gru(x)
        last_hidden_state = self.attention(h_n)
        x = self.linear1(last_hidden_state)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class attentionLSTMModel(nn.Module):
    # GRU + Self-Attention Mechanism
    def __init__(self, input_size):
        super(attentionLSTMModel, self).__init__()
        self.hidden_size = 32
        self.attention = SelfAttention(self.hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size=32, bias=False, batch_first=True, dropout=0.2)
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Use only the last hidden state
        output, h_n = self.lstm(x)
        last_hidden_state = self.attention(h_n)
        x = self.linear1(last_hidden_state)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, input_size=30, num_channels=[32, 16, 8], kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10,
                                   num_channels=num_channels,
                                   kernel_size=kernel_size,
                                   dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        tcn_output = self.tcn(x)
        # 将TCN的输出转换为(batch_size, num_channels[-1])
        tcn_output = tcn_output[:, :, -1]
        return self.linear(tcn_output)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size,  # 创建模型实例
                 hidden_size=128,  # LSTM隐藏层的大小
                 num_layers=2,  # LSTM层的数量
                 num_classes=1,  # 输出类别的数量
                 dropout_rate=0.5):  # Dropout比率)
        super(CNNLSTMModel, self).__init__()
        # 卷积层参数
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # LSTM层参数
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        # 需要将x的维度变换为卷积层期望的形状: (batch_size, input_size, seq_length)
        x = x.permute(0, 2, 1)
        # 通过卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # 将卷积层的输出调整回LSTM期望的输入形状: (batch_size, seq_length, features)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # 应用dropout
        x = self.dropout(x)
        # 选择LSTM的最后一个时间步的输出
        x = x[:, -1, :]
        # 通过全连接层
        x = self.fc(x)
        return x
class CNNGRUModel(nn.Module):
    def __init__(self, input_size,  # 创建模型实例
                 hidden_size=128,  # LSTM隐藏层的大小
                 num_layers=2,  # LSTM层的数量
                 num_classes=1,  # 输出类别的数量
                 dropout_rate=0.5):  # Dropout比率)
        super(CNNGRUModel, self).__init__()
        # 卷积层参数
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # LSTM层参数
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        # 需要将x的维度变换为卷积层期望的形状: (batch_size, input_size, seq_length)
        x = x.permute(0, 2, 1)
        # 通过卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # 将卷积层的输出调整回LSTM期望的输入形状: (batch_size, seq_length, features)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        # 应用dropout
        x = self.dropout(x)
        # 选择LSTM的最后一个时间步的输出
        x = x[:, -1, :]
        # 通过全连接层
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class FEDformerModel(nn.Module):
    def __init__(self,
                 input_dim,
                 embed_size=512,
                 num_heads=8,
                 output_dim=1,
                 max_len=10,  # 序列长度,
                 freq_feature_dim=128):  # 频域特征维度
        super(FEDformerModel, self).__init__()
        self.embed = nn.Linear(input_dim, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, max_len)
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.freq_feature_extractor = nn.Linear(max_len, freq_feature_dim)  # 简化的频域特征提取
        self.fc_out = nn.Linear(embed_size + freq_feature_dim, output_dim)  # 融合频域特征

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)
        attention_output = self.attention(x, x, x)

        # 简化的频域特征提取步骤
        freq_features = self.freq_feature_extractor(x.mean(dim=2))

        # 融合自注意力输出和频域特征
        combined_features = torch.cat((attention_output.mean(dim=1), freq_features), dim=1)
        output = self.fc_out(combined_features)
        return output


class InformerModel(nn.Module):
    def __init__(self, input_dim,
                 embed_size=512,
                 num_heads=8,
                 output_dim=1,
                 max_len=10,  # 序列长度,
                 ):
        super(InformerModel, self).__init__()
        self.embed = nn.Linear(input_dim, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, max_len)
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.fc_out = nn.Linear(embed_size, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)
        x = self.attention(x, x, x)
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.sigmoid(x))


class TemporalFusionTransformerModel(nn.Module):
    def __init__(self, input_dim,
                 embed_size=512,
                 num_heads=8,
                 output_dim=1,
                 max_len=10):  # 序列长度,
        super(TemporalFusionTransformerModel, self).__init__()
        self.embed = nn.Linear(input_dim, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, max_len)
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.glu = GatedLinearUnit(embed_size)
        self.fc_out = nn.Linear(embed_size, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)
        x = self.attention(x, x, x)
        x = self.glu(x)
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_features):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=2,
            dim_feedforward=num_features * 4, # forward_expansion
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(num_features, 1)  # 预测 1 日收益率

    def forward(self, src):
        x = self.transformer_encoder(src)
        x = x[:, -1, :]  # 取序列最后一个时间步的输出
        return self.fc_out(x)


class iTransformerModel(nn.Module):

    def __init__(self, num_features, num_layers=2, dropout_rate=0.1):
        super(iTransformerModel, self).__init__()
        self.num_features = num_features

        self.dropout = nn.Dropout(dropout_rate)

        self.positional_encodings = nn.Parameter(torch.randn(10, num_features))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=num_features,
            dim_feedforward=num_features * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(num_features)
        self.fc_out = nn.Linear(num_features, 1)  # 预测 1 日收益率

        # Dropout (self.dropout): 在应用 Transformer 编码器之前使用，用于减少训练过程中的过拟合。
        # 位置编码 (self.positional_encodings): 这个参数向模型添加了位置信息，有助于模型理解输入数据中的时间依赖性。
        # 编码器层 (self.encoder_layer 和 self.transformer_encoder): 定义了 Transformer 编码器的主体，其中使用了 2 个注意力头和四倍的前馈扩展。
        # 层归一化 (self.layer_norm): 应用于编码器层输入之前，帮助改善训练过程中的数值稳定性和收敛速度。

    def forward(self, src):
        src += self.positional_encodings[:src.size(1)]
        src = self.layer_norm(src)
        src = self.dropout(src)
        x = self.transformer_encoder(src)
        x = x[:, -1, :]
        # 取序列最后一个时间步的输出
        return self.fc_out(x)




'''
model = GRUModel(input_size=40)
print(summary(model, (30, 40)))

model = bidirectionalGRUModel(input_size=40)
print(summary(model, (30, 7)))

model = GRU2Model(input_size=40)
print(summary(model, (30, 7)))

model = attentionGRUModel(input_size=40)
print(summary(model, (30, 7)))
'''
