
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    '''
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(self.src_embed(src), src_mask)
        return self.decoder(memory, src_mask, tgt, tgt_mask)

class OnlyDecoder(nn.Module):
    '''
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    '''
    def __init__(self, decoder, tgt_embed, generator):
        super(OnlyDecoder, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, memory, tgt, src_mask, tgt_mask):
        #memory = self.encoder(self.src_embed(src), src_mask)
        return self.decoder(memory, src_mask, tgt, tgt_mask)



def clones(module, N):
    return nn.ModuleList([ copy.deepcopy(module) for _ in range(N) ])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features_size))
        self.shift = nn.Parameter(torch.zeros(features_size))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return (x - mean) * self.scale / (std + self.eps) + self.shift

class SubLayerConnection(nn.Module):
    def __init__(self, features_size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(features_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SubLayerConnection(size, dropout), 2)
        
    def forward(self, x, mask):
        attn_function = lambda x: self.self_attn(x, x, x, mask)
        x = self.sublayers[0](x, attn_function)
        return self.sublayers[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    '''
    query, key and value contain vectors corresponding to each word in the input
    '''
    sqrt_d_k = math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2,-1)) / sqrt_d_k
    
    print(scores.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    prob_scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(prob_scores)
    
    scaled_value = torch.matmul(prob_scores, value)
    return scaled_value, prob_scores

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        print(mask.shape)
        print(query.shape)
        print(key.shape)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

#class MultiHeadedAttention(nn.Module):
#    def __init__(self, num_heads, dim_input=512, dropout=0.1):
#        super(MultiHeadedAttention, self).__init__()
#        assert dim_input % num_heads == 0
#        self.num_heads = num_heads
#        self.dropout = nn.Dropout(p=dropout)
#        self.d_k = dim_input // num_heads
#        
#        # L1, L2, L3 and W0: four linear layers in all
#        self.linears = clones(nn.Linear(dim_input, dim_input), 4)
#        
#        # this is used to store the prob_scores, just for visualization
#        self.attn = None
#        
#        # helper function to resize the tensor as described above
#        self.resize_tensor = lambda tensor: tensor.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
#        
#    def forward(self, query, key, value, mask=None):
#        if mask is not None:
#            mask = mask.unsqueeze(1) # same mask is applied to all heads
#        batch_size = query.size(0)
#        
#        # use the first three linear layers to transform query, key and value
#        zipped = zip(self.linears, (query, key, value))
#        query, key, value = [self.resize_tensor(linear(x)) for (linear, x) in zipped]
#        
#        # apply self attention
#        scaled_value, self.attn = attention(query, key, value, mask, self.dropout)
#        scaled_value = scaled_value.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
#        return self.linears[-1](scaled_value)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, enc_dec_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        # enc_dec_attn is also called src_attn in the harvardnlp implementation
        self.self_attn = self_attn
        self.enc_dec_attn = enc_dec_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SubLayerConnection(size, dropout), 3)
        # we need to store size because it is used by LayerNorm in Decoder
        self.size = size
        
    def forward(self, x, encoder_outputs, src_mask, tgt_mask):
        # encoder_outputs are also called `memory` in the paper
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.enc_dec_attn(x, encoder_outputs, encoder_outputs, src_mask))
        return self.subayers[2](x, self.feed_forward)

class Generator(nn.Module):
    '''Linear + Softmax generation step'''
    def __init__(self, d_model, vocab_len):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_len)
        
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size=512, output_size=2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        positional_encodings = torch.zeros(max_len, dim_embedding)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        
        # calculate the arguments for sin and cos functions
        scale = -(math.log(10000) / dim_embedding)
        arguments = torch.arange(0, dim_embedding, 2).float() * scale
        arguments = torch.exp(arguments)
        arguments = positions * arguments
        
        # define the encodings here
        positional_encodings[:, 0::2] = torch.sin(arguments)
        positional_encodings[:, 1::2] = torch.cos(arguments)
        
        positional_encodings = positional_encodings.unsqueeze(0)
        self.register_buffer('positional_encodings', positional_encodings)
        
    def forward(self, x):
        pos_enc = self.positional_encodings[:, :x.size(1)]
        pos_enc.requires_grad_(False)
        x  = x + pos_enc
        return self.dropout(x)

class Embedding(nn.Module):
    def __init__(self, vocab_size, dim_embedding):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim_embedding)
        self.scale = math.sqrt(dim_embedding)
        
    def forward(self, x):
        # embedding is multiplied by scale to make the positional encoding relatively smaller
        # See: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
        return self.embed(x) * self.scale


def make_model(src_vocab, tgt_vocab, num_enc_dec=6, dim_model=512, dim_feedfwd=2048, attn_heads=8, dropout=0.1):
    # prepare the embeddings for encoder and decoder stacks
    position_embeddings = PositionalEncoding(dim_model, dropout)
    src_embed = nn.Sequential(Embedding(src_vocab, dim_model), copy.deepcopy(position_embeddings))
    tgt_embed = nn.Sequential(Embedding(tgt_vocab, dim_model), copy.deepcopy(position_embeddings))
    
    # prepare reusable layers. we will copy.deepcopy them whenever needed
    attn_layer = MultiHeadedAttention(attn_heads, dim_model)
    feed_fwd_layer = PositionWiseFeedForward(dim_model, dim_feedfwd, dropout)
    c = copy.deepcopy
    
    # prepare the encoder stack
    encoder_layer = EncoderLayer(dim_model, c(attn_layer), c(feed_fwd_layer), dropout)
    encoder = Encoder(encoder_layer, num_enc_dec)
    
    # prepare the decoder stack
    decoder_layer = DecoderLayer(dim_model, c(attn_layer), c(attn_layer), c(feed_fwd_layer), dropout)
    decoder = Decoder(decoder_layer, num_enc_dec)
    
    # prepare the generator
    generator = Generator(dim_model, tgt_vocab)
    
    # creat the model
    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
    
    # Initialize parameters using Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model

def make_model_news(tgt_vocab, num_enc_dec=6, dim_model=512, dim_feedfwd=2048, attn_heads=8, dropout=0.1):
    # prepare the embeddings for encoder and decoder stacks
    position_embeddings = PositionalEncoding(dim_model, dropout)
    #src_embed = nn.Sequential(Embedding(src_vocab, dim_model), copy.deepcopy(position_embeddings))
    tgt_embed = nn.Sequential(Embedding(tgt_vocab, dim_model), copy.deepcopy(position_embeddings))
    
    # prepare reusable layers. we will copy.deepcopy them whenever needed
    attn_layer = MultiHeadedAttention(attn_heads, dim_model)
    feed_fwd_layer = PositionWiseFeedForward(dim_model, dim_feedfwd, dropout)
    c = copy.deepcopy
    
    # prepare the encoder stack
    #encoder_layer = EncoderLayer(dim_model, c(attn_layer), c(feed_fwd_layer), dropout)
    #encoder = Encoder(encoder_layer, num_enc_dec)
    
    # prepare the decoder stack
    decoder_layer = DecoderLayer(dim_model, c(attn_layer), c(attn_layer), c(feed_fwd_layer), dropout)
    decoder = Decoder(decoder_layer, num_enc_dec)
    
    # prepare the generator
    generator = Generator(dim_model, tgt_vocab)
    
    # creat the model
    model = OnlyDecoder(decoder, tgt_embed, generator)
    
    # Initialize parameters using Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model


#def subsequent_mask(size):
#    "Mask out subsequent positions."
#    attn_shape = (1, size, size)
#    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#    return torch.from_numpy(subsequent_mask) == 0

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, trg=None, pad=0):
        #self.src = src
        #self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


