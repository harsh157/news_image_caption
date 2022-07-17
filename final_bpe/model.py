
import torch.nn.functional as F
class Seq2Seq(nn.Module):
    def __init__(self,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx):
        super().__init__()

        #self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = 0
        self.trg_pad_idx = 0
        #self.device = device
        self.l1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def make_src_mask(self, src):
        # src = [batch size, src len]
        #print('self.src_pad_idx:', self.src_pad_idx)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        #print('self.trg_pad_idx:', self.trg_pad_idx)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg_mask, trg, imgs):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        #imgs = imgs.view(imgs.size(0), -1, imgs.size(3))
        #imgs = self.relu(self.dropout(self.l1(imgs)))

        src_mask = self.make_src_mask(src)
        src_mask = None
        #ref_mask = self.make_src_mask(reference)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        #enc_src = self.encoder(src, src_mask, imgs)
        #enc_ref = self.encoder(reference, ref_mask, imgs) 

        outputs = []
        output, attention = self.decoder(trg, src, trg_mask, src_mask, imgs)
        # enc_src = [batch size, src len, hid dim]
        # for i in range(trg.size(1) - 1):
        #   it = trg[:, i].clone()
        #   print(it.shape)
        #   output, attention = self.decoder(it, src, trg_mask, src_mask, imgs)
        #   output = F.log_softmax(output)
        #   outputs.append(output)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        #return output, attention, src_mask
        return output
        #return torch.cat([_.unsqueeze(1) for _ in outputs], 1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=38):
        super().__init__()

        #self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        #self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.scale = np.sqrt(hid_dim)
        self.l1 = nn.Linear(hid_dim*3, 1)
        self.l2 = nn.Linear(hid_dim * 3, 1)


    def forward(self, trg, src, trg_mask, src_mask, imgs):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        #index1 = src
        #index2 = reference

        #print('imgs:', imgs.size())
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        trg1 = trg

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, trg_a, trg_v, trg_r, attention, attention2= layer(trg, src, trg_mask, src_mask, imgs)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)

        #output1 = trg2
        #print(output.shape)
        output = F.log_softmax(output, dim=2)

        # output = [batch size, trg len, output dim]

        '''
        attention = torch.mean(attention, dim=1)
        attention2 = torch.mean(attention2, dim=1)

        index1 = index1.expand(attention.size(1), index1.size(0), index1.size(1)).permute(1,0,2)
        attn_value = torch.zeros([output.size(0), output.size(1), output.size(2)]).to(device)
        attn_value = attn_value.scatter_add_(2, index1, attention)
        #attn_value = F.log_softmax(attn_value, dim=2)
        #print('trg1:', trg1.size())
        #print('output1:', output1.size())
        p = torch.sigmoid(self.l1(torch.cat([trg1, trg_a, trg_v], dim=2)))

        index2 = index2.expand(attention2.size(1), index2.size(0), index2.size(1)).permute(1, 0, 2)
        attn_value1 = torch.zeros([output.size(0), output.size(1), output.size(2)]).to(device)
        attn_value1 = attn_value1.scatter_add_(2, index2, attention2)
        # attn_value = F.log_softmax(attn_value, dim=2)
        # print('trg1:', trg1.size())
        # print('output1:', output1.size())
        q = torch.sigmoid(self.l2(torch.cat([trg1, trg_r, trg_v], dim=2)))
        output = (1 - p - q) * output + p * attn_value + q * attn_value1
        '''



        return output, output


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        #self.enc_attn_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention1 = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        #self.encoder_attention2 = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.v = nn.Linear(512, 512)
        #self.v1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()

    def forward(self, trg, src, trg_mask, src_mask, imgs):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # self attention
        #print('imgs:', imgs.size())
        imgs = self.relu(self.v(imgs))
        #print('imgs:', imgs.size())
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        # 这儿加东西
        _trg0, attention = self.encoder_attention(trg, src, src, src_mask)
        #_trg2, attention2 = self.encoder_attention2(trg, enc_ref, enc_ref, ref_mask)
        _trg1, attention1 = self.encoder_attention1(trg, imgs, imgs)

        # dropout, residual connection and layer norm
        trg1_ = _trg0
        trg2_ = _trg1
        #trg3_ = _trg2
        trg3_ = None

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg0) + self.dropout(_trg1))
        #trg = self.enc_attn_layer_norm(trg + self.dropout(_trg0) + self.dropout(_trg1) + self.dropout(_trg2))
        #trg1 = self.enc_attn_layer_norm1(trg + self.dropout(_trg1))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
      
        #return trg, trg1_, trg2_, trg3_, attention, attention2
        return trg, trg1_, trg2_, trg3_, attention, None
