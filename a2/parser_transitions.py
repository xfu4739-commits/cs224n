#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2024-2025: Homework 2
parser_transitions.py: 完成 partial parse 的转移算法
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""

import sys

class PartialParse(object):
    '''实现partial parse的数据结构与一次动作更新'''
    def __init__(self, sentence):
        """初始化一个 partial parse。

        @param sentence (list of str): 待解析句子（词列表）。
                                       你的代码不应修改该句子对象。
        """
        # 为了记录（bookkeeping）目的保留正在解析的句子。请不要在代码里修改它。
        self.sentence = sentence

        ### YOUR CODE HERE (3 Lines)
        ### 你的代码需要初始化以下字段：
        ###     self.stack: 当前栈，用 list 表示；栈顶是 list 的最后一个元素。
        ###     self.buffer: 当前缓冲区（buffer），用 list 表示；buffer 的第一个元素在 list 的最前面。
        ###     self.dependencies: 目前已生成的依存关系列表。用 (head, dependent) 形式的 tuple 表示。
        ###                        该列表的顺序不影响正确性。
        ###
        ### Note: ROOT 词元用字符串 "ROOT" 表示
        ### Note: 若需要用 sentence 来初始化字段，请避免直接引用 sentence 对象本身（例如拷贝一份），
        ###       从而确保不会修改原 sentence。
        self.stack = ["ROOT"]
        self.buffer = list(sentence) # 浅拷贝或者使用列表表达式
        self.dependencies = []
        ### END YOUR CODE


    def parse_step(self, transition):
        """对当前 partial parse 应用一次转移动作（parse step）。
        一旦确定主从关系，小弟就要从stack中移除，因为他已经找到归属

        @param transition (str): 取值为 "S"、"LA" 或 "RA"，分别表示 shift、left-arc、right-arc。
                                 你可以假设传入的 transition 一定是合法的。
        """
        ### YOUR CODE HERE (~7-12 Lines)
        ### TODO:
        ###     实现一次 parsing step，即按 PDF 讲义中的规则实现：
        ###         1) Shift
        ###         2) Left Arc
        ###         3) Right Arc
        if transition == "S":
            # 把buffer中的第一个词移到stack中
            self.stack.append(self.buffer.pop(0))
        elif transition == "LA":
            # 把stack中的第二个词作为主语，第一个词作为从属词，加入依赖关系
            self.dependencies.append((self.stack[-1], self.stack[-2]))
            self.stack.pop(-2)
        elif transition == "RA":
            # 把stack中的第一个词作为主语，第二个词作为从属词，加入依赖关系
            self.dependencies.append((self.stack[-2], self.stack[-1]))
            self.stack.pop(-1)
        ### END YOUR CODE

    def parse(self, transitions):
        """将给定的 transitions 依次应用到该 `PartialParse` 上。

        @param transitions (list of str): 按顺序要执行的转移序列

        @return dependencies (list of (str, str)): 解析完成后生成的依存关系列表，
                                                   每个元素为 (head, dependent)
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """用 minibatch 的方式解析一组句子。

    @param sentences (list[list[str]]): 待解析句子列表（每个句子是词的列表）
    @param model (ParserModel): 用于做解析决策的模型。假设其实现了 `model.predict(partial_parses)`，
                                输入是 `PartialParse` 列表，输出是每个 parse 的下一步 transition 列表。
                                即：
                                    transitions = model.predict(partial_parses)
                                那么 `transitions[i]` 就是要应用到 `partial_parses[i]` 的下一步动作。
    @param batch_size (int): 每个 minibatch 中包含的 `PartialParse` 数量

    @return dependencies (list[list[tuple]]): 返回列表中每个元素是对应句子的依存关系列表。
                                              顺序应与 `sentences` 一致（即 `dependencies[i]` 对应 `sentences[i]`）。
    """
    dependencies = []

    ### YOUR CODE HERE (~8-10 Lines)
    ### TODO:
    ###     实现 minibatch parse 算法。该算法的伪代码见 PDF 讲义。
    ###
    ###     Note: 浅拷贝（PDF 中所述）在 Python 中可以用切片/赋值实现，例如：
    ###                 unfinished_parses = partial_parses[:].
    ###             这里 `unfinished_parses` 是 `partial_parses` 的浅拷贝。
    ###             在 Python 中，浅拷贝得到的新 list 并不会创建新对象实例；两个 list 只是引用同一批对象。
    ###             本题里 `partial_parses` 存放的是 `PartialParse` 对象列表，`unfinished_parses` 也引用同一批对象。
    ###             因此不要使用 `del` 从 `unfinished_parses` 中删除对象；这可能释放底层内存，导致 `partial_parses`
    ###             访问到无效对象并引起崩溃。
    partial_parses = [PartialParse(s) for s in sentences]  # 创建初始的partial parse对象
    unfinished_parses = partial_parses[:] # 创建未完成的partial parse对象

    while unfinished_parses:
        minibatch = unfinished_parses[:batch_size] # 创建小批量
        transitions = model.predict(minibatch) # 预测小批量的下一步动作

        for pp, t in zip(minibatch, transitions):
            pp.parse_step(t) # 应用小批量的下一步动作   
        # 移除已经完成的partial parse对象
        unfinished_parses = [pp for pp in unfinished_parses
                             if not (len(pp.buffer) == 0 and len(pp.stack) == 1)] # 更新未完成的partial parse对象

    dependencies = [pp.dependencies for pp in partial_parses] # 获取所有完成的partial parse对象的依赖关系
    ### END YOUR CODE
    return dependencies # 返回所有完成的partial parse对象的依赖关系


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """测试一次 parse_step 是否得到期望输出"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print("{:} test passed!".format(name))


def test_parse_step():
    """`PartialParse.parse_step` 的简单测试
    注意：这些测试并不全面
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """`PartialParse.parse` 的简单测试
    注意：这些测试并不全面
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    """用于测试 `minibatch_parse` 的虚拟模型（Dummy model）"""
    def __init__(self, mode = "unidirectional"):
        self.mode = mode

    def predict(self, partial_parses):
        if self.mode == "unidirectional":
            return self.unidirectional_predict(partial_parses)
        elif self.mode == "interleave":
            return self.interleave_predict(partial_parses)
        else:
            raise NotImplementedError()

    def unidirectional_predict(self, partial_parses):
        """先把所有词 shift 到栈上；随后如果句子的第一个词是 "right" 就只做 right-arc，否则只做 left-arc。"""
        return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

    def interleave_predict(self, partial_parses):
        """先把所有词 shift 到栈上；随后交替进行 "right" 与 "left" 的 arc 动作。"""
        return [("RA" if len(pp.stack) % 2 == 0 else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

def test_dependencies(name, deps, ex_deps):
    """测试依存关系是否与期望一致"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """`minibatch_parse` 的简单测试
    注意：这些测试并不全面
    """

    # 单向 arc 测试
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))

    # 越界测试
    sentences = [["right"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0], (('ROOT', 'right'),))

    # 混合 arc 测试
    sentences = [["this", "is", "interleaving", "dependency", "test"]]
    deps = minibatch_parse(sentences, DummyModel(mode="interleave"), 1)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'is'), ('dependency', 'interleaving'),
                      ('dependency', 'test'), ('is', 'dependency'), ('is', 'this')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_minibatch_parse()
    else:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
