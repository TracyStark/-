from model.model import Encoder
from model.model import DecoderWithAttention
from config import *
import torch

encoder = Encoder()
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


test_params_flop(encoder, (1, 64, 64))
test_params_flop(decoder, (512, 32, 32))
