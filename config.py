# 数据路径
dataset_dir = "data/MyDataset"  # 更新为正确的目录
vocab_path = 'data/MyDataset/vocab_v3.txt'
train_set_path = './data/small/train.json'
val_set_path = './data/small/val.json'

# 模型参数
emb_dim = 512  # 词嵌入维数
attention_dim = 512  # attention 层维度
decoder_dim = 512  # decoder维度
dropout = 0.5
buckets = [[240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
           [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
           [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
           [1000, 400], [1200, 200], [1600, 200],
           ]

# 训练参数
start_epoch = 0
epochs = 20  # 不触发早停机制时候最大迭代次数
epochs_since_improvement = 0  # 用于跟踪在验证集上分数没有提高的迭代次数
batch_size = 32  # 训练集批大小
test_batch_size = 32  # 验证集批大小
encoder_lr = 1e-4  # 编码器学习率
decoder_lr = 4e-4  # 解码器学习率
grad_clip = 5.  # 梯度裁剪阈值
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_score = 0.  # 目前最好的 score 
print_freq = 100  # 状态的批次打印间隔
checkpoint = None  # checkpoint文件目录(用于断点继续训练)
save_freq = 2  # 保存的间隔
