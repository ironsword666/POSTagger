import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from src.modules.char_lstm import CharLSTM
from src.modules.dropout import SharedDropout, IndependentDropout

class Tagger_Model(nn.Module):
    '''
    Model(
    (word_embedding): Embedding(23144, 100)
    (bilstm): LSTM(200, 400, num_layers=3, batch_first=True, bidirectional=True)
    (fc): Linear()
    )
    '''

    def __init__(self, 
                 n_words, 
                 n_feats,
                 n_tags, 
                 feat='char',
                 n_embed=100, 
                 n_char_embed=50,
                 n_feat_embed=100,
                 embed_dropout=.33,
                 n_lstm_hidden=400, 
                 n_lstm_layer=1, 
                 lstm_dropout=.33,
                 pad_index=0, 
                 unk_index=1,
                 feat_pad_index=0):
        super(Tagger_Model, self).__init__()

        self.pad_index = pad_index
        self.unk_index = unk_index
        
        # Embedding Layer
        self.word_embedding = nn.Embedding(num_embeddings=n_words,
                                           embedding_dim=n_embed)

        # TODO charlstm and bert embedding

        if feat == 'char':
            # charlstm
            self.feat_embedding = CharLSTM(n_chars=n_feats,
                                           n_char_embed=n_char_embed,
                                           n_out=n_feat_embed,
                                           pad_index=feat_pad_index)

        elif feat == 'bert':
            raise NotImplementedError('bert feature hasn\'t been implemented !')

        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # LSTM Layer
        self.bilstm = nn.LSTM(input_size=n_embed+n_feat_embed,
                              hidden_size=n_lstm_hidden,
                              num_layers=n_lstm_layer,
                              batch_first=True,
                              bidirectional=True)
        
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        
        # TODO transform? add a layer transform n_lstm_hidden*2 to n_lstm_hidden then activate

        # Linear Layer
        self.linear = nn.Linear(in_features=n_lstm_hidden*2,
                                out_features=n_tags)

        # state transition matrix
        # TODO initialize
        self.transition = nn.Parameter(torch.randn(n_tags, n_tags))

    def forward(self, words, feats):
        '''
        Params:
            words (Tensor(batch_size, seq_len): ...
            feats (Tensor(batch_size, seq_len, fix_len)): ...
        '''

        # words not padded, mask: Tensor(batch, seq_len)
        mask = words.ne(self.pad_index)
        # actual length of sequence, lens: Tensor(batch)
        lens = mask.sum(dim=1)

        # Embedding Layer

        # find words whose index is beyond word_embedding boundry
        outside_mask = words.ge(self.word_embedding.num_embeddings)
        # replace outside indices with unk tag, now, all words are inside boundry
        inside_words = words.masked_fill(outside_mask, self.unk_index)
        # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        word_embed = self.word_embedding(inside_words)

        # pretrained embedding
        if hasattr(self, 'pretrained_embedding'):
            # (batch, seq_len, embedding_dim) + (batch, seq_len, embedding_dim)
            word_embed += self.pretrained_embedding(words)

        # we can also use chars[mask] to get Tensor(sum(actual_seq_len), fix_len)
        # feat_embed = self.char_lstm(chars[mask])
        feat_embed = self.feat_embedding(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # (batch, seq_len, embedding_dim*2)
        embed = torch.cat((word_embed, feat_embed), dim=-1)

        # BiLSTM Layer

        x = pack_padded_sequence(embed, lens, batch_first=True, enforce_sorted=False)
        x, _ = self.bilstm(x)
        # (batch, seq_len, n_lstm_hidden*2)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.lstm_dropout(x)
        
        # Linear Layer

        # (batch, seq_len, n_lstm_hidden*2) -> (batch, seq_len, tag_nums)
        # seq_len include <bos> <eos> and <pad>
        scores = self.linear(x)

        # there is no need to mask tags, because tokens can emit to each tag 

        return scores

    def load_pretrained(self, embed=None):
        '''load pretrained embeddings.

        Params:
            embed Tensor(*,*): pretrained embeddings
        '''

        if embed is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(embed)
            ## static or not?
            # self.pretrained_embedding.weight.requires_grad = False



      
        


