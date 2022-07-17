
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.functional import log_softmax, pad

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

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
        
    def forward(self, memory, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        #print(tgt.shape)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

#class DecoderLayer(nn.Module):
#    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
#
#    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#        super(DecoderLayer, self).__init__()
#        self.size = size
#        self.self_attn = self_attn
#        self.src_attn = src_attn
#        self.feed_forward = feed_forward
#        self.sublayer = clones(SublayerConnection(size, dropout), 3)
#
#    def forward(self, x, memory, src_mask, tgt_mask):
#        "Follow Figure 1 (right) for connections."
#        m = memory
#        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
#        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
#        return self.sublayer[2](x, self.feed_forward)


class GehringLinear(nn.Linear):
    """A linear layer with Gehring initialization and weight normalization."""

    def __init__(self, in_features, out_features, dropout=0, bias=True,
                 weight_norm=True):
        self.dropout = dropout
        self.weight_norm = weight_norm
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        # One problem with initialization from the uniform distribution is that
        # the distribution of the outputs has a variance that grows with the
        # number of inputs. It turns out that we can normalize the variance of
        # each neuron’s output to 1 by scaling its weight vector by the square
        # root of its fan-in (i.e. its number of inputs). Dropout further
        # increases the variance of each input, so we need to scale down std.
        # See A.3. in Gehring et al (2017): https://arxiv.org/pdf/1705.03122.
        std = math.sqrt((1 - self.dropout) / self.in_features)
        self.weight.data.normal_(mean=0, std=std)
        if self.bias is not None:
            self.bias.data.fill_(0)

        # Weight normalization is a reparameterization that decouples the
        # magnitude of a weight tensor from its direction. See Salimans and
        # Kingma (2016): https://arxiv.org/abs/1602.07868.
        if self.weight_norm:
            nn.utils.weight_norm(self)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, img_attn, article_attn, feed_forward, gh_linear, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.img_attn = img_attn
        self.article_attn = article_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 4)
        self.context_fc = gh_linear

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        img = memory['image']
        article = memory['article']
        article_mask = memory['article_mask']
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        #TODO Key masking in MHAttention
        x_img = self.sublayer[1](x, lambda x: self.img_attn(x, img, img, src_mask))
        x_article = self.sublayer[2](x, lambda x: self.article_attn(x, article, article, article_mask))

        X_context = torch.cat([x_img, x_article], dim=-1)
        x = self.context_fc(X_context)

        return self.sublayer[3](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        #print(scores.size())
        #print(mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_key, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_key // h
        self.d_q = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.linear_kv = clones(nn.Linear(d_key, d_model),2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linears[0](query).view(nbatches, -1, self.h, self.d_q).transpose(1, 2)
        key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_q).transpose(1, 2)
            for lin, x in zip(self.linear_kv, (key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_q)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, sparse=False):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, sparse=sparse)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

#def make_model(src_vocab, tgt_vocab, num_enc_dec=6, dim_model=512, dim_feedfwd=2048, attn_heads=8, dropout=0.1, img_dim=512, sent_dim=512):
#    # prepare the embeddings for encoder and decoder stacks
#    position_embeddings = PositionalEncoding(dim_model, dropout)
#    src_embed = nn.Sequential(Embedding(src_vocab, dim_model), copy.deepcopy(position_embeddings))
#    tgt_embed = nn.Sequential(Embedding(tgt_vocab, dim_model), copy.deepcopy(position_embeddings))
#    
#    # prepare reusable layers. we will copy.deepcopy them whenever needed
#    self_attn_layer = MultiHeadedAttention(attn_heads, dim_model, dim_model)
#    img_attn_layer = MultiHeadedAttention(attn_heads, dim_model, img_dim)
#    sent_attn_layer = MultiHeadedAttention(attn_heads, dim_model, sent_dim)
#    feed_fwd_layer = PositionWiseFeedForward(dim_model, dim_feedfwd, dropout)
#    c = copy.deepcopy
#    
#    # prepare the encoder stack
#    encoder_layer = EncoderLayer(dim_model, c(attn_layer), c(feed_fwd_layer), dropout)
#    encoder = Encoder(encoder_layer, num_enc_dec)
#    
#    # prepare the decoder stack
#    decoder_layer = DecoderLayer(dim_model, c(attn_layer), c(attn_layer), c(feed_fwd_layer), dropout)
#    decoder = Decoder(decoder_layer, num_enc_dec)
#    
#    # prepare the generator
#    generator = Generator(dim_model, tgt_vocab)
#    
#    # creat the model
#    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
#    
#    # Initialize parameters using Xavier initialization
#    for p in model.parameters():
#        if p.dim() > 1:
#            nn.init.xavier_uniform_(p)
#            
#    return model

def make_model_news(tgt_vocab, num_enc_dec=6, dim_model=512, dim_feedfwd=2048, attn_heads=8, dropout=0.1, img_dim=2048, sent_dim=300):
    # prepare the embeddings for encoder and decoder stacks
    position_embeddings = PositionalEncoding(dim_model, dropout)
    #src_embed = nn.Sequential(Embedding(src_vocab, dim_model), copy.deepcopy(position_embeddings))
    tgt_embed = nn.Sequential(Embeddings( dim_model, tgt_vocab), copy.deepcopy(position_embeddings))
    gh_linear = GehringLinear(dim_model*2, dim_model, weight_norm=False)

    
    # prepare reusable layers. we will copy.deepcopy them whenever needed
    attn_layer = MultiHeadedAttention(attn_heads, dim_model, dim_model)
    img_attn_layer = MultiHeadedAttention(attn_heads, dim_model, img_dim)
    sent_attn_layer = MultiHeadedAttention(attn_heads, dim_model, sent_dim)
    feed_fwd_layer = PositionwiseFeedForward(dim_model, dim_feedfwd, dropout)
    c = copy.deepcopy
    c(gh_linear)
    # prepare the encoder stack
    #encoder_layer = EncoderLayer(dim_model, c(attn_layer), c(feed_fwd_layer), dropout)
    #encoder = Encoder(encoder_layer, num_enc_dec)
    
    # prepare the decoder stack
    decoder_layer = DecoderLayer(dim_model, c(attn_layer), c(img_attn_layer), c(sent_attn_layer), c(feed_fwd_layer), c(gh_linear), dropout)
    decoder = Decoder(decoder_layer, num_enc_dec)
    
    # prepare the generator
    generator = Generator(dim_model, tgt_vocab)
    
    # creat the model
    model = OnlyDecoder(decoder, tgt_embed, generator)
    
    # Initialize parameters using Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in generator.parameters():
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


