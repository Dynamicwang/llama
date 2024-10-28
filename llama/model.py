# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        # torch.rsqrt 计算平方根的倒数 is the same as 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    # 将输入张量reshape为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 将频率reshape为xq的shape
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 计算复数乘法，然后进行维度的变换
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # 计算结果返回
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 将输入的x张量进行广播，然后reshape成新的形状,即把最后最后一个维度复制n_rep次
    # 然后再将复制扩充后的数据reshape成新的形状
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        # 
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        #
        model_parallel_size = fs_init.get_model_parallel_world_size()
        #
        self.n_local_heads = args.n_heads // model_parallel_size
        
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads
        # 查询的线性变换, 其中的权重大小为（dim, n_heads * head_dims）
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # 键的线性变换 ,其中的权重大小为（dim, kv_heads * head_dim）
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # 值的线性变换 ,其中的权重大小为（dim, kv_heads * head_dim）
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # 输出的线性变换 ,其中的权重大小为（n_heads * head_dims, dim）
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # cache的大小为（max_batch_size, max_seq_len, kv_heads, head_dim）
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        # 得到batch size和序列长度
        bsz, seqlen, _ = x.shape
        # 计算得到Q, K, V
        # Q: (batchsize, seq_len, args.n_heads * self.head_dim)
        # K: (batchsize, seq_len, n_kv_heads * self.head_dim)
        # V: (batchsize, seq_len, n_kv_heads * self.head_dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # reshape Q, K, V的形状，将head和dim拆分
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 将Q, K 进行旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # 将kvcache拷贝到xq所在的设备上
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        # 计算的K， V存储到相应的cache内部

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        # 从cache中取出K， V
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # 存储本次的kv,是从start_pos到start_pos+seqlen
        # 在上一步中，是从0开始到start_pos+seqlen，那么认为0：start_pos是cache中已经有的kv，即为cache_len
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # 将head维度转换为第二个维度, 方便后面以头为单位进行计算
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # 计算QK点积，并除以根号下head_dim，得到注意力分数， 注意K需要转置
        # 根据维度信息，scores的维度为(bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # mask的维度为（seqlen, seqlen+start_pos）==> (seqlen, cache_len + seqlen)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # 计算softmax，得到注意力权重
        # 使得输出向量的所有元素都在 (0, 1) 之间，并且所有元素的和为 1。这使得 softmax 函数非常适合用于表示分类任务中的类别概率
        # exp(x)/sum(exp(x))，使得每个元素都变成了概率分布
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 计算注意力权重与value的乘积，得到输出向量
        # score的维度为：（bs, n_local_heads, seqlen, start_pos+seqlen）
        # values的维度为：（bs, n_local_heads, cache_len+seqlen, head_dim）
        # output的维度为(bs, n_local_heads, seqlen, head_dim)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # 将多头注意力输出拼接起来，得到最终的输出向量
        # output的维度为（bs, seqlen, n_local_heads * head_dim）
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # 通过线性变换得到最终的输出向量
        # output的维度为（bs, seqlen, dim）
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        # 计算隐藏层的维度
        # hidden_dim的入参是 4 * dim, 
        # hidden_dim = int(8 * dim / 3)
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        # 通过两个因子，计算hidden_dim
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    # 前向传播的逻辑，
    # 输入x，经过w1, silu激活函数，经过w3,
    # 将w1和w3的结果相乘，逐元素相乘，
    # 最后经过w2
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        # 多头
        self.n_heads = args.n_heads
        # 嵌入后的数据长度
        self.dim = args.dim
        # 得到每个头的长度
        self.head_dim = args.dim // args.n_heads
        # 初始化注意力机制
        self.attention = Attention(args)
        # 初始化前馈神经网络
        # dim为数据嵌入后的长度
        # hidden_dim为隐藏层的长度为4 * dim
        # multiple_of
        # ffn_dim_multiplier
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        # 初始化层id
        self.layer_id = layer_id
        # 初始化层归一化
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 初始化前馈归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # 对x进行norm 操作然后再进行attention，最后再与x相加,类似resnet的操作
        # attention后得到的结果仍然为x的维度，（bs, seqlen, dim）
        # h的维度为（bs, seqlen, dim）
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        # 对h进行norm操作然后再进行feedforward，最后再与h相加， 类似resnet的操作
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


# llama的模型结构
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        # 词表大小
        self.vocab_size = params.vocab_size
        # 模型层数
        self.n_layers = params.n_layers

        # 词嵌入层，使用可并行的嵌入层
        # 词嵌入层的意思是：将一个token映射为一个向量
        # token：[xxxx, token id, xxx, ....] 列表的大小为vocab_size, 输出维度为dim
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        # Transformer层，使用一个List存储
        self.layers = torch.nn.ModuleList()
        # 根据外部参数，构建Transformer层
        for layer_id in range(params.n_layers):
            # 每一层为一个TransformerBlock， 参数为layer_id和params
            self.layers.append(TransformerBlock(layer_id, params))
        # 
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 模型输出层， 使用可并行的线性层， 输入维度为dim，输出维度为vocab_size
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        # 位置编码
        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # 根据输入的token,获取对应的batch size和序列长度
        _bsz, seqlen = tokens.shape
        # 根据token获取对应的嵌入向量
        h = self.tok_embeddings(tokens)
        # 将位置编码拷贝到和h相同的设备上
        self.freqs_cis = self.freqs_cis.to(h.device)
        # 获取特定位置的编码
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            # 将mask的形状设置为和tokens相同的方阵形状， 值为负无穷大
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )
            # mask对角线上以及对角线以下的值设置为0
            # diagonal：控制对角线上的偏移量，默认为0，表示对角线以下的设置为0，对角线和以上值保持不变
            # 如果diagonal为正， 表示在对脚线向上的偏移量
            # 如果diagonal为负， 表示在对脚线向下的偏移量
            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            # mask的形状为(seqlen, start_pos + seqlen)
            # 0, ...., 0, -inf, -inf, ...
            # 0, ...., 0, 0   , -inf, ...
            # 0, ...., 0, 0   , 0   ,...
            # .......
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # 将结果进行归一化
        h = self.norm(h)
        # 输出层， 将结果映射到vocab_size维度
        output = self.output(h).float()
        return output
