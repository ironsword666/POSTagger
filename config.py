import os
import sys

class Config(object):
    '''神经网络的配置参数'''

    # [网络]
    max_len = 50             # 文本的最大长度
    
    n_embed = 100          # 词向量的维度
    n_char_embed=50         # char embedding dim
    n_feat_embed=100        # 和词向量拼接的特征
    n_lstm_layer = 2             # layers of bilstm
    n_lstm_hidden = 400       # lstm的hidden层

    min_freq = 2                # 低频词频率
    fix_len = 20            # 每个单词最多有多少个字符  
    

    # [优化系数]
    lr = 2e-3        # 学习率
    # l2_reg_lambda = 0.01        # 正则化系数
    mu = 0.9
    nu = 0.9
    epsilon = 1e-12
    clip = 5              # 梯度截断上限
    decay = 0.75              # 学习率下降
    decay_steps = 5000

    # [训练设置]
    batch_size = 50             # 每批训练的大小
    epochs = 50            # 总迭代轮次
    patience = 10           # dev上多少次不提升则停止训练

    # # [数据]
    # base_dir = './data/ptb/'                   # 数据文件所在的目录
    # train_file = os.path.join(base_dir, 'train.conllx')     # 训练集
    # dev_file = os.path.join(base_dir, 'dev.conllx')         # 验证集
    # test_file = os.path.join(base_dir, 'test.conllx')       # 测试集
    # word2vec_file = './data/embedding/glove.6B.100d.txt' # word2vec

    # model_file = './save/model/model_save'

    # pretrained_embedding_file = './save/embedding/pretrained.pt'  # word2vec数据部分

    # Comand
    # python run.py --word2vec_file=./data/embedding/glove.6B.100d.txt --unk=unk