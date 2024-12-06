import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    # dim为1表示在时间维度T上进行FFT计算，xf的形状仍是[B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    # abs(xf)计算出每个频率成分的幅值
    # .mean(0)代表在第一个维度即B维度上进行平均，目的为了去除样本之间差异，张量形状从[B, T, C]变成[T, C]
    # .mean(-1)代表在最后一个维度即C维度上平均，这一步是在不同特征之间取平均，的带在所有特征之间的平均幅值。张量形状从[T, C]变成了[T]
    # 最终结果得到了包含T个频率未知的平均幅值序列，每个位置的值代表整个batch和所有特征上的总体幅值特征
    frequency_list = abs(xf).mean(0).mean(-1)
    # 直流分量是指信号在频率为零时的成分，我们更关心周期的变化，period = T // f （f != 0），设置0防止频率为0的值被top_k选中
    frequency_list[0] = 0
    # _是返回topK寻找到的最大幅值，top_list是找到值的索引位置，使用_的意思是忽略掉第一个返回值只保留位置信息
    _, top_list = torch.topk(frequency_list, k)
    # 将 top_list 从计算图中分离（不再跟踪梯度），并转移到CPU上，然后转换为NumPy数组，便于后续处理。
    top_list = top_list.detach().cpu().numpy()
    # 计算周期长度，就是每个频率对应的周期长度
    period = x.shape[1] // top_list
    # 返回包含top_k个主频率的周期长度
    # abs(xf).mean(-1)[:, top_list] ： 每条数据在top_k个主频率的幅值
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
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
        # period_list:周期长度,一维数值
        # period_weight: [B, k] 每个值代表的是在某个Batch上的幅值，
        # period_weight计算幅值时候是将batch平均在一起后找top_k幅值，现在返回的是那top_K中每个batch的具体幅值
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                # 计算padding尺寸，填充在时间步T上
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                # 将创建的padding向量添加在时间维度T上
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            # 这句话目的是让N成为第二维度，N也可理解为channels，为了满足2D卷积的输入格式，即【batch_size,channels,height,width】Height代表行数、width代表列数
            # out形状 B, N, length of period, period
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back, similar to reshape
            # out.permute的形状 B, length of period, period, N
            # .reshape形状 B, T, N
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # truncating down the padded part of the output and put it to result
            # 截断out里填充部分填充到res里面，res初始值是空
            # out[:, :(self.seq_len + self.pred_len), :], B的数据全要, T从0开始截止self.seq_len + self.pred_len，N全要
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        # res = [tensor1, tensor2, ..., tensor_k]  # 每个 tensor 的形状为 (B, T, N)
        # 调用 torch.stack(res, dim=-1) 后，这些张量将沿着最后一个维度堆叠，形成一个新的张量，形状变为：(B, T, N, k)
        # torch.stack 就是增加一个新维度
        res = torch.stack(res, dim=-1) #res: 4D [B, length , N, top_k]
        # adaptive aggregation
        #First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
        # 计算每个batch的分布概率
        period_weight = F.softmax(period_weight, dim=1)
        #after two unsqueeze(1),shape -> [B,1,1,top_k],so repeat the weight to fit the shape of res
        # unsqueeze(1) 在第1维度上增加维度
        # repeat(1, T, N, 1)，沿着指定的维度重复张量的元素
        # 第一个1是0维不变，仍然是B
        # T，在第一维度上重复T次
        # N，在第二维度上重复M次
        # 最后那个1是最后一维值不变
        # 最后period_weight变成(B, T, N, k) 与res一样，res是经过2维卷积处理过的，period_weight存的幅值
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        #add by weight the top_k periods' result, getting the result of this TimesBlock
        # 最后一个维度相乘相加，变成(B, T, N)，也是论文里对tok根据幅值加权求和
        res = torch.sum(res * period_weight, -1)
        # residual connection
        # 残差连接
        res = res + x
        return res


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
