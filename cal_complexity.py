from model.model import ResNetEncoder
from model.model import DecoderWithAttention
from config import *
import torch

encoder = ResNetEncoder()
decoder = DecoderWithAttention(attention_dim=attention_dim,
                               embed_dim=emb_dim,
                               decoder_dim=decoder_dim,
                               vocab_size=414,
                               dropout=dropout)

def test_params_flop(model, x_shape):
    """
    You need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.cuda(),
            x_shape,
            as_strings=True,
            print_per_layer_stat=False
        )
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# 计算编码器的参数和 FLOPs
test_params_flop(encoder, (3, 224, 224))  # 修改输入形状为 (3, 224, 224)

# 计算解码器的参数和 FLOPs
# 解码器的输入形状应该与编码器的输出形状一致
encoder_output_shape = (2048, 7, 7)  # 假设编码器输出的形状为 (batch_size, 2048, 7, 7)
decoder_input_shape = (1, 2048, 7, 7)  # 添加 batch_size 维度
test_params_flop(decoder, decoder_input_shape)
