import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from layers.SelfAttention_Family import AttentionLayer, FullAttention


def FFT_for_Period(x, k=2):
    B, _, C = x.shape
    # [B, T, C]
    # dim为1表示在时间维度T上进行FFT计算，xf的形状仍是[B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    # abs(xf)计算出每个频率成分的幅值
    # .mean(0)代表在第一个维度即B维度上进行平均，目的为了去除样本之间差异，张量形状从[B, T, C]变成[T, C]
    # 的到[T,C]的张量，每列代表一个变量
    # 最终结果得到了包含T个频率未知的平均幅值序列，每个位置的值代表整个batch和所有特征上的总体幅值特征
    frequency_list = abs(xf).mean(0)
    # 直流分量是指信号在频率为零时的成分，我们更关心周期的变化，period = T // f （f != 0），设置0防止频率为0的值被top_k选中
    frequency_list[0, :] = 0
    # _是返回topK寻找到的最大幅值，top_list是找到值的索引位置，使用_的意思是忽略掉第一个返回值只保留位置信息
    # top_list 形状[topK, C]; 存的是索引位置 
    _, top_list = torch.topk(frequency_list, k, dim=0)
    # 将 top_list 从计算图中分离（不再跟踪梯度），并转移到CPU上，然后转换为NumPy数组，便于后续处理。
    top_list = top_list.detach().cpu().numpy()
    # 计算周期长度，就是每个频率对应的周期长度
    period = x.shape[1] // top_list
    # period_list:周期长度,二维数值[topK,N]
    # abs(xf).mean(-1)[:, top_list] ： [B, k, N] 每个值代表的是在某个Batch上的幅值，
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        # 添加多头注意力部分
        self.attention_block = TransformerBlock(
            d_model=configs.d_model,
            n_heads=configs.d_model, 
            d_ff=configs.d_ff,
            dropout=configs.dropout
        )
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        # period_list: [topK, N] 每个变量的周期长度
        # period_weight: [B, k, N] 每个batch中每个变量的幅值
        period_list, period_weight = FFT_for_Period(x, self.k)
        
        res = []
        for i in range(self.k):
            # 经过自注意力机制处理各变量之间关系
            # x_attended = self.attention_block(x.permute(0,2,1)).permute(0,2,1)
            x_attended = x
            # 对每个变量使用其对应的周期进行处理
            variable_res = []
            for j in range(N):
                period = period_list[i, j]  # 获取第i个topk中第j个变量的周期
                
                # 获取当前变量的数据
                x_var = x_attended[:, :, j:j+1]  # [B, T, 1]
                
                # padding
                if (self.seq_len + self.pred_len) % period != 0:
                    length = (((self.seq_len + self.pred_len) // period) + 1) * period
                    padding = torch.zeros([B, (length - (self.seq_len + self.pred_len)), 1]).to(x_var.device)
                    out = torch.cat([x_var, padding], dim=1)
                else:
                    length = self.seq_len + self.pred_len
                    out = x_var
                
                # reshape 为2D形式处理
                # [B, 1, length//period, period]
                
                out = out.reshape(B, length // period, period, 1).permute(0, 3, 1, 2).contiguous()
               
                out = out.repeat(1,N,1,1)
                # 2D conv处理
                
                out = self.conv(out)
                
                # out = out.sum(dim=-1, keepdim=True)
                # reshape回原来的形式
                out = out.permute(0, 2, 3, 1).reshape(B, -1, 1)
                
                # 截断填充的部分
                out = out[:, :(self.seq_len + self.pred_len), :]
                variable_res.append(out)
            
            # 将所有变量的结果拼接在一起
            # variable_res中每个张量形状为[B, T, 1]，拼接后变为[B, T, N]
            period_res = torch.cat(variable_res, dim=2)
            res.append(period_res)
        
        # 将k次的结果堆叠 [B, T, N, k]
        res = torch.stack(res, dim=-1)
        
        # 使用period_weight进行加权聚合
        period_weight = F.softmax(period_weight, dim=1)  # [B, k, N]
        # 调整period_weight的形状以匹配res
        period_weight = period_weight.permute(0, 2, 1)  # [B, N, k]
        period_weight = period_weight.unsqueeze(1)  # [B, 1, N, k]
        period_weight = period_weight.repeat(1, T, 1, 1)  # [B, T, N, k]
        
        # 加权求和
        res = torch.sum(res * period_weight, -1)  # [B, T, N]
        
        # 残差连接
        res = res + x
        return res

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = AttentionLayer(
            attention=FullAttention(mask_flag=False, attention_dropout=dropout),
            d_model=d_model,
            n_heads=n_heads
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=None)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        # 归一化层，防止梯度爆炸或梯度消失
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            #  nn.Linear()，变化维度，将最后一个维度特征数层第一个变为第二个特征数
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # 标准化 代码基本固定
        means = x_enc.mean(1, keepdim=True).detach() #[B, T]
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # 将T放到最后一个维度进行线性预测然后再换回[B, T, C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        # 反标准化 代码基本固定
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
