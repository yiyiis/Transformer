import torch
import math
import copy
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        item = 1/1000**(torch.arange(0, d_model, 2)/d_model)

        tmp_pos = position * item

        pe[:, 0::2] = torch.sin(tmp_pos)
        pe[:, 1::2] = torch.cos(tmp_pos)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe
        x = x+pe[:, x.shape[1], :]
        return self.dropout(x)
    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]

    scores = torch.matmul(query, key.transpose(-2, -1)/math.sqrt(d_k))

    if mask is not None:
        # print("scores", scores.shape)
        scores = scores.masked_fill(mask, 1e-9)

        

    p_attn = F.softmax(scores, dim=-1)

    # print(p_attn.sum(dim=-1))
    # print(p_attn)

    
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % head == 0

        self.d_k = embedding_dim // head

        self.head = head

        self.embedding_dim = embedding_dim

        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # if mask is not None:
        #     mask = mask.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.head, 1, 1)


        batch_size = query.shape[0]
        # print("query",query.shape)

        query, key, value = [ model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for model, x in zip(self.linears, (query, key, value)) ]

        # print("query",query.shape)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # print(self.attn[0][0])
        # print(self.attn)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # print(self.a2.shape)
        # print(x.shape)
        # print('mean:' , mean.shape)
        # print('std', std.shape)
        return self.a2*(x-mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        self.size = size

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        

        return self.sublayer[2](x, self.feed_forward)
    

# 解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_target_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_target_mask, target_mask)

        return self.norm(x)
    
# 分类
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        print("-------------!")
        return F.log_softmax(self.project(x), dim=-1)


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, source_embed, target_embed, pad_idx, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.pad_idx = pad_idx
        self.generator = generator


    def forward(self, source, target):
        return self.generator(self.decode(self.encode(source, self.generate_mask(source, source)), self.generate_mask(target, source), target, self.generate_mask(target, target, True)))

    def encode(self, source, source_mask):
        return self.encoder(self.source_embed(source), source_mask)

    def decode(self, memory, source_target_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, source_target_mask, target_mask)
    
    def generate_mask(self, query, key, is_triu_mask=False):

        device = query.device
        batch, seq_q = query.shape
        batch, seq_k = key.shape

        mask = (key==self.pad_idx).unsqueeze(1)
        mask = mask.expand(batch, seq_q, seq_k).to(device)

        if is_triu_mask:
            dst_triu_mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool))
            dst_triu_mask = dst_triu_mask.unsqueeze(0).expand(batch, seq_q, seq_k)
            return mask|dst_triu_mask
        
        return mask

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, pad_idx):
        super(Embeddings, self).__init__()
        self.em = nn.Embedding(vocab, d_model, pad_idx)

    def forward(self, x):
        return self.em(x)
    

def make_model(source_vocab, target_vocab, pad_idx, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab, pad_idx), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab, pad_idx), c(position)),
        pad_idx,
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

    
if __name__ == "__main__":
    model = make_model(100, 200, 0)
    a = torch.LongTensor([[1, 2, 0, 0],
    [1, 2, 3, 0]])
    b = torch.LongTensor([[2, 3, 4, 0, 0],
    [1, 2, 3, 0, 0]])
    result = model(a, b)
    print(result.shape)
    # pass



