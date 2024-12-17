import time
from config import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from model.utils import *
from model import metrics, dataloader, model
from torch.utils.checkpoint import checkpoint as train_ck
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.dataloader import MyDataset
import matplotlib.pyplot as plt

test_path = './data/MyDataset_test/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.device = device

if checkpoint is None:
    print("no checkpoint found")
    exit(1)
else:
    print(checkpoint)
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_score = checkpoint['score']
    decoder = checkpoint['decoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']

decoder = decoder.to(device)
encoder = encoder.to(device)

test_dataset = MyDataset(test_path, is_train=False, is_test=True)

test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_MyDataset, num_workers=0)

decoder.eval()

if encoder is not None:
    encoder.eval()

words = []
with open(vocab_path) as f:
    words = f.read().splitlines()
words.append("<start>")
words.append("<end>")

with tqdm(enumerate(test_loader), leave=False, total=len(test_loader), position=0) as it:
    for i, (imgs, caps, caplens) in it:
        print(imgs.shape)
        imgs = imgs.to(device)
        if encoder is not None:
            imgs = encoder(imgs)
            predictions, alphas,the_list = decoder.inference(imgs)
            word_list = []
            for i in the_list:
                word_list.append(words[i])
            with open("output_2.txt", "a") as file:
                file.write(" ".join(word_list))  # 使用逗号分隔列表元素
                file.write("\n")
