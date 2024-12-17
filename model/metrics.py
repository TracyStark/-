import numpy as np
import distance
from nltk.translate.bleu_score import sentence_bleu

def evaluate(losses, top3accs, references, hypotheses):
    """
    用于在验证集上计算各种评价指标指导模型早停
    :param losses: 损失值
    :param top3accs: top-3 准确率
    :param references: 参考答案
    :param hypotheses: 模型预测结果
    :return: 包含所有指标的字典
    """
    # Calculate scores
    bleu4 = 0.0
    for i, j in zip(references, hypotheses):
        bleu4 += max(sentence_bleu([i], j), 0.01)
    bleu4 = bleu4 / len(references)
    exact_match = exact_match_score(references, hypotheses)
    edit_distance = edit_distance_score(references, hypotheses)
    score = bleu4 + exact_match + edit_distance / 10
    print(
        '\n * LOSS:{loss.avg:.3f},TOP-3 ACCURACY:{top3.avg:.3f},BLEU-4:{bleu:.3f},Exact Match:{exact_match:.1f},Edit Distance:{edit_distance:.3f},Score:{score:.6f}'.format(
            loss=losses,
            top3=top3accs,
            bleu=bleu4,
            exact_match=exact_match,
            edit_distance=edit_distance,
            score=score))
    return {
        'BLEU-4': bleu4,
        'Exact Match': exact_match,
        'Edit Distance': edit_distance,
        'Score': score
    }

def exact_match_score(references, hypotheses):
    """
    Computes exact match scores.
    :param references: list of list of tokens (one ref)
    :param hypotheses: list of list of tokens (one hypothesis)
    :return: exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))

def edit_distance_score(references, hypotheses):
    """
    Computes Levenshtein distance between two sequences.
    :param references: list of list of token (one hypothesis)
    :param hypotheses: list of list of token (one hypothesis)
    :return: 1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot