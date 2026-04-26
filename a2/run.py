#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2023-2024: Homework 2
run.py: 运行依存句法分析器
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
from datetime import datetime
import os
import pickle
import math
import time
import argparse

from torch import nn, optim
import torch
from tqdm import tqdm

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
parser.add_argument('-d', '--debug', action='store_true', help='是否进入 debug 模式（使用更小的数据子集）')
args = parser.parse_args()

# -----------------
# Primary Functions
# -----------------
def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    """训练神经网络依存句法分析器（dependency parser）。

    @param parser (Parser): 神经依存句法分析器
    @param train_data (): 训练数据
    @param dev_data (): 开发集数据
    @param output_path (str): 保存模型权重与结果的路径
    @param batch_size (int): 每个 batch 的样本数
    @param n_epochs (int): 训练轮数
    @param lr (float): 学习率
    """
    best_dev_UAS = 0 #记录最好的开发集UAS


    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###      1) 构造 Adam 优化器并赋值给变量 `optimizer`
    ###      2) 构造交叉熵损失函数并赋值给变量 `loss_func`，使用 `mean` 作为 reduction（默认）
    ###
    ### Hint: 用 `parser.model.parameters()` 作为优化器需要更新的参数。
    ### 相关文档：
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(reduction='mean')

    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            # todo
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """训练一个 epoch。

    注意：在 PyTorch 中，通过 `model.train()` 与 `model.eval()` 可以切换训练/评估模式，
    从而自动启用或禁用 Dropout 等层的行为。

    @param parser (Parser): 神经依存句法分析器
    @param train_data (): 训练数据
    @param dev_data (): 开发集数据
    @param optimizer (nn.Optimizer): Adam 优化器
    @param loss_func (nn.CrossEntropyLoss): 交叉熵损失函数
    @param batch_size (int): batch size

    @return dev_UAS (float): 开发集的 Unlabeled Attachment Score (UAS)
    """
    parser.model.train() # 切到训练模式（例如启用 dropout）
    # TODO 最后一个batch可能不够batch_size，需要特殊处理
    n_minibatches = math.ceil(len(train_data) / batch_size) #计算有几个batch
    loss_meter = AverageMeter() #累计平均loss，最后打印

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            #pytorch的梯度是累加的，不清零就会把上一个batch的梯度叠加进来
            optimizer.zero_grad()   # 清空历史梯度
            loss = 0. # 当前 batch 的 loss
            # 将train_x和train_y从numpy转换为tensor
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            ### YOUR CODE HERE (~4-10 lines)
            ### TODO:
            ###      1) 将 train_x 前向通过模型得到 `logits`
            ###      2) 用传入的 `loss_func` 计算 PyTorch 的交叉熵损失。该函数以 `logits` 与 `train_y` 为输入，
            ###         输出 softmax(`logits`) 与 `train_y` 的交叉熵损失。注意 softmax(`logits`) 即 PDF 中的预测 \( \hat{y} \)。
            ###      3) 反向传播（backprop）
            ###      4) 调用优化器执行一步更新
            ### 相关文档：
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            # 前向传播
            logits = parser.model(train_x) #(batch_size, n_classes)
            # 计算损失
            loss = loss_func(logits, train_y) #(batch_size,)
            # 反向传播
            loss.backward()
            # todo 更新参数，每一个batch都会更新参数 
            optimizer.step()

            ### END YOUR CODE
            # 更新进度条，前进一个batch
            prog.update(1)
            # 更新累计平均loss，loss是tensor，需要转换为标量
            loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("在 dev 集上评估",)
    parser.model.eval() # 切到评估模式（例如禁用 dropout）
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    debug = args.debug

    assert (torch.__version__.split(".") >= ["1", "0", "0"]), "Please install torch version >= 1.0.0"

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print("耗时 {:.2f} 秒\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("加载在 dev 集上表现最好的模型权重")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
