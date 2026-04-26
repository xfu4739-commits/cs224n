#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2024-2025: Homework 2
parser_model.py: 用于依存句法分析的前馈神经网络
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
    """带 embedding 层与两层隐层的前馈神经网络。

    `ParserModel` 用于在给定部分解析（partial parse）配置时，预测下一步应该应用哪一种转移（transition）。

    PyTorch 备注：
        - `ParserModel` 继承自 `nn.Module`。在 PyTorch 中，所有神经网络模型通常都继承自该基类。
        - `__init__` 方法用于定义网络中的层与参数（embedding、线性层、dropout 等）。
        - 当你创建类的新实例时（例如 `m = ParserModel(...)`），`__init__` 会被自动调用。
        - 类中的其他方法可以访问带 `self.` 前缀的成员变量；因此需要在 `__init__` 中把你希望后续使用的层/参数保存为 `self.xxx`。
        - 更多 `nn.Module` 文档参见：https://pytorch.org/docs/stable/nn.html
    """
    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        """初始化 parser 模型。

        @param embeddings (ndarray): 词向量矩阵 (num_words, embedding_size)
        @param n_features (int): 输入token数量，后面forward会传入的token数量
        @param hidden_size (int): 隐层单元数
        @param n_classes (int): 输出类别数
        @param dropout_prob (float): dropout 概率
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1] # 词向量维度
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings)) # TODO 放到GPU上

        ### YOUR CODE HERE (~9-10 Lines)
        ### TODO:
        ###     1) 将 `self.embed_to_hidden_weight` 与 `self.embed_to_hidden_bias` 声明为 `nn.Parameter`。
        ###        用 `nn.init.xavier_uniform_` 初始化 weight，用 `nn.init.uniform_`（默认参数）初始化 bias。
        ###     2) 构造 `self.dropout` 层。
        ###     3) 将 `self.hidden_to_logits_weight` 与 `self.hidden_to_logits_bias` 声明为 `nn.Parameter`。
        ###        用 `nn.init.xavier_uniform_` 初始化 weight，用 `nn.init.uniform_`（默认参数）初始化 bias。
        ###
        ### Note: 可训练变量通常声明为 `nn.Parameter`，这样张量会被纳入计算图中，并能根据梯度进行更新。
        ###       这里对 weight 使用 Xavier Uniform 初始化。经验上它往往比随机均匀初始化更利于网络训练。
        ###       更多解释可参考这篇博客：
        ###             http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        ###
        ### 相关文档：
        ###     nn.Parameter: https://pytorch.org/docs/stable/nn.html#parameters
        ###     Initialization: https://pytorch.org/docs/stable/nn.init.html
        ###     Dropout: https://pytorch.org/docs/stable/nn.html#dropout-layers
        ### 
        ### 提示详见作业 PDF。

        input_size = self.n_features * self.embed_size # 输入token数量 * 词向量维度，拼接成一个向量
        # nn.parameter 是pytorch中的一个类，用于将张量包装成模型参数，从而可以被自动求导，requires_grad=True
        # torch.empty 是pytorch中的一个函数，用于创建一个未初始化的张量
        # nn.init.xavier_uniform_ 是pytorch中的一个函数，用于初始化权重，使得权重服从均匀分布
        # nn.init.uniform_ 是pytorch中的一个函数，用于初始化偏置，使得偏置服从均匀分布
        self.embed_to_hidden_weight = nn.Parameter(torch.empty(input_size, self.hidden_size))
        nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        self.embed_to_hidden_bias = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.uniform_(self.embed_to_hidden_bias)

        self.dropout = nn.Dropout(p=self.dropout_prob)

        self.hidden_to_logits_weight = nn.Parameter(torch.empty(self.hidden_size, self.n_classes))
        nn.init.xavier_uniform_(self.hidden_to_logits_weight)
        self.hidden_to_logits_bias = nn.Parameter(torch.empty(self.n_classes))
        nn.init.uniform_(self.hidden_to_logits_bias)

        ### END YOUR CODE

    def embedding_lookup(self, w):
        """使用 `w` 从 embedding 矩阵 `self.embeddings` 中取出对应的向量。

            @param w (Tensor): 输入的词索引张量 (batch_size, n_features)

            @return x (Tensor): 与 `w` 中各索引对应的 embedding，形状为
                                (batch_size, n_features * embed_size)
        """

        ### YOUR CODE HERE (~1-4 Lines)
        ### TODO:
        ###     1) 对 `w` 中每个索引 `i`，从 `self.embeddings` 选出第 `i` 个向量
        ###     2) 如有需要，使用 `view` 对张量进行 reshape
        ###
        ### Note: 所有 embedding 向量按行堆叠存成一个矩阵。模型接收的是一串词的索引，
        ###       然后通过本函数把索引映射为对应的 embedding 序列。
        ###
        ###       本题旨在考察你对 embedding lookup 的理解，因此不要使用诸如 `nn.Embedding`
        ###       这样的高层 API（题目希望你自己实现它）。请特别注意张量的形状，必要时进行 reshape。
        ###       在运行代码前，务必明确每个张量的 shape。
        ###
        ### PyTorch 有一些有用的 API（除了 `nn.Embedding` 之外）可用于实现本题，相关文档如下：
        ###     Index select: https://pytorch.org/docs/stable/torch.html#torch.index_select
        ###     Gather: https://pytorch.org/docs/stable/torch.html#torch.gather
        ###     View: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Flatten: https://pytorch.org/docs/stable/generated/torch.flatten.html
        x = None
        x = torch.index_select(self.embeddings, 0, w.view(-1)) #(batch_size * n_features, embed_size)
        x = x.view(w.size(0), -1)   #(batch_size, n_features * embed_size)

        ### END YOUR CODE
        return x


    def forward(self, w):
        """前向传播（forward）。

            注意：这里不会显式应用 softmax，因为损失函数 `nn.CrossEntropyLoss` 内部已经包含了更高效的实现。

            PyTorch 备注：
                - 每个 `nn.Module`（PyTorch 模型）都有一个 `forward` 函数。
                - 当你用输入张量 `w` 调用模型时，会执行该 `forward`。
                    例如：
                        model = ParserModel(...)
                        output = model(w)  # 这会调用 forward
                - 更多细节参见：https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward

        @param w (Tensor): 输入 token 索引张量 (batch_size, n_features)

        @return logits (Tensor): 网络输出的 logits（未做 softmax），形状为 (batch_size, n_classes)
        """
        ### YOUR CODE HERE (~3-5 lines)
        ### TODO:
        ###     按讲义（write-up）描述完成 forward 计算；并在 ReLU 之后加入在 `__init__` 中声明的 dropout 层。
        ###
        ### Note: 不在此处对 logits 做 softmax；因为损失函数（`torch.nn.CrossEntropyLoss`）会以更高效的方式完成。
        ###
        ### 相关文档：
        ###     Matrix product: https://pytorch.org/docs/stable/torch.html#torch.matmul
        ###     ReLU: https://pytorch.org/docs/stable/nn.html?highlight=relu#torch.nn.functional.relu
        logits = None
        # 1. 线性层
        x = self.embedding_lookup(w)  # (batch_size, n_features * embed_size)
        # 2. 隐藏层
        h = torch.matmul(x, self.embed_to_hidden_weight) + self.embed_to_hidden_bias  # (batch_size, hidden_size)
        h = F.relu(h) # 3. ReLU 激活函数
        h = self.dropout(h) # 4. Dropout 层
        # 5. 输出层
        logits = torch.matmul(h, self.hidden_to_logits_weight) + self.hidden_to_logits_bias  # (batch_size, n_classes)

        ### END YOUR CODE
        return logits


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple sanity check for parser_model.py')
    parser.add_argument('-e', '--embedding', action='store_true', help='对 embedding_lookup 函数做简单一致性检查')
    parser.add_argument('-f', '--forward', action='store_true', help='对 forward 函数做简单一致性检查')
    args = parser.parse_args()

    embeddings = np.zeros((100, 30), dtype=np.float32)
    model = ParserModel(embeddings)

    def check_embedding():
        inds = torch.randint(0, 100, (4, 36), dtype=torch.long)  #(4.36)
        selected = model.embedding_lookup(inds)  #(4, 36 * 30)
        # np.all 是numpy中的一个函数，用于判断数组中是否所有元素都为True
        # selected.data.numpy() 是pytorch中的一个函数，用于将张量转换为numpy数组
        # repr 是python中的一个函数，用于将对象转换为字符串
        # assert 是python中的一个语句，用于判断条件是否为True，如果为False，则抛出异常
        assert np.all(selected.data.numpy() == 0), "The result of embedding lookup: " \
                                      + repr(selected) + " contains non-zero elements."

    def check_forward():
        """
        检查forward函数的输出形状是否正确
        """
        inputs =torch.randint(0, 100, (4, 36), dtype=torch.long)
        out = model(inputs)
        expected_out_shape = (4, 3)
        assert out.shape == expected_out_shape, "The result shape of forward is: " + repr(out.shape) + \
                                                " which doesn't match expected " + repr(expected_out_shape)

    if args.embedding:
        check_embedding()
        print("Embedding_lookup sanity check passes!")

    if args.forward:
        check_forward()
        print("Forward sanity check passes!")
