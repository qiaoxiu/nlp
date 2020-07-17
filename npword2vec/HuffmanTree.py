__author__ = 'multiangle'
# 这是实现 霍夫曼树相关的文件， 主要用于 针对层次softmax进行 word2vec 优化方案的一种
'''
至于 为什么要进行层次softmax  可以简单理解 因为词表很大 针对上完个类别单词进行softmax 计算量大 更新参数过多 无法训练，而采用softmax 层次化  只需要 计算几个有限单词的sigmod 就可以 更新参数也非常少
提高训练速度

什么是霍夫曼树 简单理解就是 将训练文本 进行词频统计 通过构建加权最短路径来构造二叉树 这样 词频高的 位置在前 词频低的位置在后 每一个 霍夫曼编码代表一个词 路径 并且是唯一 不是其他词的前缀

'''
import numpy as np

class HuffmanTreeNode():
    def __init__(self,value,possibility):
        # common part of leaf node and tree node
        # 词频概率，训练文本出现的次数
        self.possibility = possibility
        # 左右子节点
        self.left = None
        self.right = None
        # value of leaf node  will be the word, and be
        # mid vector in tree node
        # 叶节点是学习的词向量  非叶子节点是中间变量 即 wx 与 xite
        self.value = value # the value of word
        # 存储霍夫曼码
        self.Huffman = "" # store the huffman code

    def __str__(self):
        return 'HuffmanTreeNode object, value: {v}, possibility: {p}, Huffman: {h}'\
            .format(v=self.value,p=self.possibility,h=self.Huffman)

class HuffmanTree():
    def __init__(self, word_dict, vec_len=15000):
        self.vec_len = vec_len      # the length of word vector
        self.root = None
        # 所有词汇
        word_dict_list = list(word_dict.values())
        # 根据所有词汇信息 创建节点
        node_list = [HuffmanTreeNode(x['word'],x['possibility']) for x in word_dict_list]
        # 构建霍夫曼树
        self.build_tree(node_list)
        # self.build_CBT(node_list)
        # 生成霍夫曼树的霍夫曼编码
        self.generate_huffman_code(self.root, word_dict)

    def build_tree(self,node_list):
        # node_list.sort(key=lambda x:x.possibility,reverse=True)
        # for i in range(node_list.__len__()-1)[::-1]:
        #     top_node = self.merge(node_list[i],node_list[i+1])
        #     node_list.insert(i,top_node)
        # self.root = node_list[0]

        while node_list.__len__()>1:
            i1 = 0  # i1表示概率最小的节点
            i2 = 1  # i2 概率第二小的节点
            if node_list[i2].possibility < node_list[i1].possibility :
                [i1,i2] = [i2,i1]
            for i in range(2,node_list.__len__()): # 找到最小的两个节点
                if node_list[i].possibility<node_list[i2].possibility :
                    i2 = i
                    if node_list[i2].possibility < node_list[i1].possibility :
                        [i1,i2] = [i2,i1]
             #根据 叶节点1 和叶节点2 生成叶节点 也就是中间变量 其中 用来 存放xite  
            top_node = self.merge(node_list[i1],node_list[i2])
            # 删除节点1 和节点2  将 新生成的非叶节点进行 加入 以进行后续 循环构建霍夫曼树
            if i1<i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1>i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0,top_node)
        self.root = node_list[0]

    def build_CBT(self,node_list): # build a complete binary tree
        node_list.sort(key=lambda  x:x.possibility,reverse=True)
        node_num = node_list.__len__()
        before_start = 0
        while node_num>1 :
            for i in range(node_num>>1):
                top_node = self.merge(node_list[before_start+i*2],node_list[before_start+i*2+1])
                node_list.append(top_node)
            if node_num%2==1:
                top_node = self.merge(node_list[before_start+i*2+2],node_list[-1])
                node_list[-1] = top_node
            before_start = before_start + node_num
            node_num = node_num>>1
        self.root = node_list[-1]

    def generate_huffman_code(self, node, word_dict):
        # # use recursion in this edition
        # if node.left==None and node.right==None :
        #     word = node.value
        #     code = node.Huffman
        #     print(word,code)
        #     word_dict[word]['Huffman'] = code
        #     return -1
        #
        # code = node.Huffman
        # if code==None:
        #     code = ""
        # node.left.Huffman = code + "1"
        # node.right.Huffman = code + "0"
        # self.generate_huffman_code(node.left, word_dict)
        # self.generate_huffman_code(node.right, word_dict)

        # use stack butnot recursion in this edition
        # 左子树 编码是1 右子树 编码是0 先左子树 在右字数 设置编码链
        stack = [self.root]
        while (stack.__len__()>0):
            node = stack.pop()
            # go along left tree
            while node.left or node.right :
                code = node.Huffman
                node.left.Huffman = code + "1"
                node.right.Huffman = code + "0"
                stack.append(node.right)
                node = node.left
            word = node.value
            code = node.Huffman
            # print(word,'\t',code.__len__(),'\t',node.possibility)
            word_dict[word]['Huffman'] = code

    def merge(self,node1,node2):
        # 新生成的非叶节点的词频是 俩个叶节点的加和
        top_pos = node1.possibility + node2.possibility
        # 将非叶节点向量进行初始化
        top_node = HuffmanTreeNode(np.zeros([1,self.vec_len]), top_pos)
        if node1.possibility >= node2.possibility :
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node











