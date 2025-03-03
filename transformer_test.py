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
        scores = scores.masked_fill(mask ==0, -1e9)

        

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

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)

        return self.norm(x)
    
# 分类
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator


    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.source_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.em = nn.Embedding(vocab, d_model)

    def forward(self, x):
        return self.em(x)



def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


    
if __name__ == "__main__":
    # # 编码
    d_model = 512
    vocab = 1000
    x = torch.LongTensor([[100, 2, 421, 500], [2, 23, 521, 12]])
    emb = nn.Embedding(vocab, d_model)
    embr = emb(x)
    # print(embr.shape)

    # 位置编码
    dropout = 0.1
    max_len = 60
    x = embr
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(x)
    # print(pe_result.shape)

    # #注意力
    # mask = torch.zeros(2, 4, 4)
    # query = key = value = pe_result
    # attn, p_attn = attention(query, key, value, mask=mask, dropout=nn.Dropout(0.8))
    # print('attn', attn.shape)
    # print('p_attn', p_attn.shape)
    # print(p_attn)

    #多头注意力机制 
    head = 8
    embedding_dim = 512
    dropout = 0.1

    query = key = value = pe_result
    mask = torch.zeros(2, 4, 4)

    mha = MultiHeadedAttention(head, embedding_dim, dropout)
    mha_result = mha(query, key, value, mask)


    #加全连接层
    d_model = 512
    d_ff=64
    dropout=0.2

    x = mha_result
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_result = ff(x)
    # print(ff_result)
    # print(ff_result.shape)


    # 规范化
    features = d_model = 512
    eps = 1e-6

    x = ff_result
    ln = LayerNorm(features, eps)
    ln_result = ln(x)
    # print(ln_result.shape)

    # 连接子层
    size = d_model = 512
    head = 8
    dropout = 0.2
    x = pe_result
    mask = torch.zeros(2, 4, 4)
    self_attn = MultiHeadedAttention(head, d_model, dropout)

    sublayer = lambda x: self_attn(x, x, x, mask)

    sc = SublayerConnection(size, dropout)
    sc_result = sc(x, sublayer)
    # print(sc_result.shape)

    # encoderlayer
    # size = 512
    # head = 8
    # d_model = 512
    # d_ff = 64
    # x = pe_result
    # dropout = 0.2
    # self_attn = MultiHeadedAttention(head, d_model)
    # ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # mask = torch.zeros(2, 4, 4)

    # el = EncoderLayer(size , self_attn, ff, dropout)
    # el_result = el(x, mask)
    # print(el_result.shape)
    
    # encoder

    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    x = pe_result
    dropout = 0.2
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model,dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    N = 8
    mask = torch.zeros(2, 4, 4)

    layer = EncoderLayer(size, c(attn), c(ff), dropout)

    en = Encoder(layer, N)
    en_result = en(x, mask)
    # print(en_result.shape)

    # 编码器层
    head = 8
    size = 512
    d_model = 512
    d_ff = 64
    dropout = 0.2
    self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    x = pe_result
    memory = en_result

    mask = torch.zeros(2, 4, 4)

    source_mask = target_mask = mask
    dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
    dl_result = dl(x, memory, source_mask, target_mask)
    # print(dl_result.shape)

    # 解码器
    size = 512
    d_model = 512
    head = 8
    dropout = 0.2
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    N = 8

    x = pe_result
    memory = en_result
    mask = torch.zeros(2, 4, 4)
    source_mask = target_mask = mask

    de = Decoder(layer, N)
    de_result = de(x, memory, source_mask, target_mask)
    # print(de_result.shape)

    # linear
    d_model = 512
    vocab_size = 1000
    x = de_result

    gen = Generator(d_model, vocab_size)
    gen_result = gen(x)
    # print(gen_result.shape)

    # 编解码器
    vocab_size = 1000
    d_model = 512
    encoder = en
    decoder = de
    source_embed = nn.Embedding(vocab_size, d_model)
    target_embed = nn.Embedding(vocab_size, d_model)
    generator = gen

    source = target = torch.LongTensor([[1, 2, 3, 4], [2, 3, 4,5]])

    source_mask = target_mask = torch.zeros(2, 4, 4)
    ed = EncoderDecoder(encoder, decoder, target_embed, target_embed, generator)
    ed_result = ed(source, target, source_mask, target_mask)
    # print(ed_result.shape)
    # print(ed_result)

    # model
    source_vocab = 11
    target_vocab = 11
    N = 6
    res = make_model(source_vocab, target_vocab)
    print(res)
    res_result = res(source, target, source_mask, target_mask)
    print(res_result.shape)

    





