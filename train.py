import time
import pandas as pd
from tensorboardX import SummaryWriter
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
from model.dataloader import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.device = device

# cudnn.benchmark = True

def main():
    """
    Training and validation.
    """

    global best_score, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='./runs/experiment_1')

    with open(vocab_path) as f:
        words = f.readlines()
    words.append("<start>")
    words.append("<end>")
    word_map = {value.strip(): index + 1 for index, value in enumerate(words)}
    word_map["<pad>"] = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = model.DecoderWithAttention(attention_dim=attention_dim,
                                             embed_dim=emb_dim,
                                             decoder_dim=decoder_dim,
                                             vocab_size=len(word_map),
                                             dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = model.ResNetEncoder(input_channels=3)  # 确保输入通道数为3
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_score = checkpoint['score']
        decoder = checkpoint['decoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 自定义的数据集
    train_dataset = MyDataset(dataset_dir, is_train=True)
    eval_dataset = MyDataset(dataset_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_MyDataset,
        num_workers=0
    )
    val_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_MyDataset,
        num_workers=0
    )

    p = 0.06 #1  # teacher forcing概率

    # 用于记录每个 epoch 的结果
    results = []

    # Epochs
    for epoch in tqdm(range(start_epoch, epochs)):
        # 每2个epoch衰减一次teahcer forcing的概率
        if p > 0.05:
            if epoch % 3 == 0 and epoch != 0:
                p *= 0.75
        else:
            p = 0

        # 如果迭代4次后没有改善,则对学习率进行衰减,如果迭代20次都没有改善则触发早停.直到最大迭代次数
        if epochs_since_improvement == 30:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           encoder=encoder,
                           decoder=decoder,
                           criterion=criterion,
                           encoder_optimizer=encoder_optimizer,
                           decoder_optimizer=decoder_optimizer,
                           epoch=epoch, p=p)

        # One epoch's validation
        val_loss, top3_acc, bleu4, exact_match, edit_distance, score = validate(val_loader=val_loader,
                                                                                encoder=encoder,
                                                                                decoder=decoder,
                                                                                criterion=criterion)

        writer.add_scalar('Training Loss', np.array(train_loss), epoch)
        writer.add_scalar('Validation Score', np.array(score), epoch)
        writer.add_scalar('Teacher Forcing Probability', np.array(p), epoch)
        if (p == 0):
            print('Stop teacher forcing!')
            # Check if there was an improvement
            is_best = score > best_score
            best_score = max(score, best_score)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                print('New Best Score!(%d)' % (best_score,))
                epochs_since_improvement = 0

            if epoch % save_freq == 0:
                print('Saveing...')
                save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                                decoder_optimizer, score, is_best)
        print('--------------------------------------------------------------------------')

        # 记录每个 epoch 的结果
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'top3_acc': top3_acc,
            'bleu4': bleu4,
            'exact_match': exact_match,
            'edit_distance': edit_distance,
            'score': score,
            'teacher_forcing_probability': p
        })

    writer.close()

    # 保存结果到 Excel 文件
    df = pd.DataFrame(results)
    df.to_excel('training_results.xlsx', index=False)
    print('Results saved to training_results.xlsx')


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, p):
    """
    Performs one epoch's training.
    :param train_loader: 训练集的dataloader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 损失函数
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    total_loss = 0.0
    batch_count = 0

    # Batches
    with tqdm(enumerate(train_loader), total=len(train_loader), position=0) as it:
        for i, (imgs, caps, caplens) in it:
            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            try:
                imgs = encoder(imgs)
            except:
                imgs = train_ck(encoder, imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, p=p)

            # 由于加入开始符<start>以及停止符<end>,caption从第二位开始,知道结束符
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths.cpu().int(), batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths.cpu().int(), batch_first=True).data

            # Calculate loss
            scores = scores.to(device)
            loss = criterion(scores, targets)

            # 加入 doubly stochastic attention 正则化
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # 反向传播
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)

            # 更新权重
            decoder_optimizer.step()
            encoder_optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            it.set_postfix(
                Loss=f"{loss:.4e}",
                Avg_Loss=f"{total_loss / batch_count:.4e}"
            )

    return total_loss / batch_count


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: 用于验证集的dataloader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 损失函数
    :return: 验证集上的指标
    """
    decoder.eval()  # 推断模式,取消dropout以及批标准化
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top3accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        with tqdm(enumerate(val_loader), leave=False, total=len(val_loader), position=0) as it:
            for i, (imgs, caps, caplens) in it:

                # Move to device, if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)

                # Forward prop.
                if encoder is not None:
                    imgs = encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, p=0)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths.cpu(), batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths.cpu(), batch_first=True).data

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top3 = accuracy(scores, targets, 3)
                top3accs.update(top3, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                it.set_postfix(
                    Loss=f"{loss:.2e}",
                )

                # Store references (true captions), and hypothesis (prediction) for each image
                caplens = caplens[sort_ind].int()
                caps = caps[sort_ind]
                for i in range(len(caplens)):
                    end = caplens[i].int().item()
                    references.append(caps[i][1:end].tolist())
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    end = decode_lengths[j].int().item()
                    temp_preds.append(preds[j][:end])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

            Score = metrics.evaluate(losses, top3accs, references, hypotheses)
    return losses.avg, top3accs.avg, Score['BLEU-4'], Score['Exact Match'], Score['Edit Distance'], Score['Score']


if __name__ == '__main__':
    main()
