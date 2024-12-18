import torch
import numpy as np
import json
from model.model import ResNetEncoder, DecoderWithAttention
from config import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.dataloader import MyDataset

# 加载模型
encoder = ResNetEncoder()
decoder = DecoderWithAttention(attention_dim=512, embed_dim=512, decoder_dim=512, vocab_size=414, dropout=0.5)

# 加载训练好的权重
checkpoint = torch.load('BEST_checkpoint_MyDataset.pth')
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# 将模型设置为评估模式
encoder.eval()
decoder.eval()

# 加载测试数据集
test_path = './data/'
test_dataset = MyDataset(test_path, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_MyDataset, num_workers=0)

# 加载词汇表
with open('word_map.json', 'r') as j:
    word_map = json.load(j)
reverse_word_map = {v: k for k, v in word_map.items()}

# 进行推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
decoder = decoder.to(device)

with torch.no_grad():
    for imgs, caps, caplens in test_loader:
        imgs = imgs.to(device)
        encoder_out = encoder(imgs)
        h, c = decoder.init_hidden_state(encoder_out)
        predictions, alphas, inference_list = decoder.inference(encoder_out)

        # 将索引转换为单词
        predicted_words = [reverse_word_map[idx] for idx in inference_list[0]]
        print('Predicted words:', predicted_words)

        # 保存输出标签
        with open("output_labels.txt", "a") as file:
            file.write(" ".join(predicted_words))  # 使用空格分隔列表元素
            file.write("\n")