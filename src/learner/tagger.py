import sys
import os
from datetime import datetime, timedelta

from src.models.tagger_model import Tagger_Model
from src.utils.field import Field, SubWordField
from src.utils.corpus import Conll, Embedding
from src.utils.dataloader import TextDataSet
from src.utils.common import pad_token, unk_token, bos_token, eos_token
from src.utils.algorithms import crf_forward, viterbi
from src.utils.functions import preprocessing_heads
from src.utils.metric import Metric

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

class Tagger(object):

    def __init__(self, args, tri_fields, model):

        self.args = args
        self.true_fields = {k: value[0] for k, value in tri_fields.items()}
        # {field:alias, ...}
        self.alias_fields = {value[0]: value[1] for value in tri_fields.values()}
        self.tagger_model = model

    def train(self, args):
        '''
        
        Params:
            : 
        
        '''

        # build dataset
        train = TextDataSet(Conll.load(args.ftrain), self.alias_fields)
        train.build_loader(batch_size=args.batch_size, shuffle=True)
        dev = TextDataSet(Conll.load(args.fdev), self.alias_fields)
        dev.build_loader(batch_size=args.batch_size)
        test = TextDataSet(Conll.load(args.ftest), self.alias_fields)
        test.build_loader(batch_size=args.batch_size)
        
        self.criterion = nn.CrossEntropyLoss(reduction='mean') 
        # Adam Optimizer
        self.optimizer = Adam(self.tagger_model.parameters(), args.lr) 
        # learning rate decrease
        # new_lr = initial_lr * gamma**epoch = initial_lr * 0.75**(epoch/5000)
        self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps)) 

        total_time = timedelta()
        best_epoch, metric = 0, Metric()
        for epoch in range(1, args.epochs+1):
            start_time = datetime.now()

            print('training epoch {} :'.format(epoch))
            loss, metric = self.train_epoch(args, train.data_loader)
            print('train loss: {}'.format(loss))
            accuracy = self.evaluate(args, dev.data_loader)
            print('dev accuracy: {}'.format(accuracy))

            time_diff = datetime.now() - start_time
            print('epoch time: {}'.format(time_diff))
            total_time += time_diff
            # if accuracy > best_accuracy:
                # best_epoch = epoch

            # if epoch - best_epoch > args.patience:
                # break
        accuracy = self.evaluate(args, test.data_loader)
        print('test accuracy: {}'.format(accuracy))
        print('total_time: {}'.format(total_time))
    
    def train_epoch(self, args, data_loader):

        self.tagger_model.train() 
        total_loss = 0 

        for words, feats, tags in data_loader:

            self.optimizer.zero_grad()
            # mask <bos> <eos > and <pad>
            mask = words.ne(self.true_fields['WORD'].pad_index) & words.ne(self.true_fields['WORD'].bos_index) & words.ne(self.true_fields['WORD'].eos_index)
            # (batch, seq_len, tag_nums); include <bos> <eos> and <pad>
            scores = self.tagger_model(words, feats) 

            transition = None
            if args.use_crf:
                transition = self.tagger_model.transition

            loss = self.get_loss(self.criterion, scores, tags, mask, use_crf=args.use_crf, transition=transition)
            # compute grad
            loss.backward() 
            # clip grad which is larger than args.clip
            nn.utils.clip_grad_norm_(self.tagger_model.parameters(), args.clip)
            # backpropagation
            self.optimizer.step()
            # updat learning rate
            self.scheduler.step()
            total_loss += loss.item()

        # TODO metric    
        
        return total_loss / len(data_loader), 0

    @torch.no_grad()
    def evaluate(self, args, data_loader):
        ''''''
        
        self.tagger_model.eval()
        n_total, n_right = 0, 0 # 所有文本，分类对了的文本
        for words, feats, tags in data_loader:

            mask = words.ne(self.true_fields['WORD'].pad_index) & words.ne(self.true_fields['WORD'].bos_index) & words.ne(self.true_fields['WORD'].eos_index)
            scores = self.tagger_model(words, feats) 
            if args.use_crf:
                preds = viterbi(scores, mask, self.tagger_model.transition)
            else:
                preds = torch.argmax(scores, dim=-1)
            # print(words.size(), feats.size(), scores.size(), out.size(), tags.size(), mask.size())
            equal = torch.eq(preds[mask], tags[mask])
            n_right += torch.sum(equal).item()
            n_total += torch.sum(mask).item()

        print('right and total: ', n_right, n_total)
        accuracy = n_right / n_total
        return accuracy

    def get_loss(self, criterion, scores, tags, mask, use_crf=False, transition=None):
        '''
        local loss: use crossentropy and scores
        global loss: use crf, scores and transition
        
        for score matrix, we drop out illegal tokens, that is <bos> and <pad>
        for each score vector, we just save scores that exceed sentence_len, but -inf will have no effect
        a score vector: [1., 4., 1., 0., 4., 3., 2., -inf]
        
        we split every token, not treat them as a part of sentences, that is we flat a batch of sentences.

        Params:
            criterion: loss function
            scores (Tensor(batch, seq_len, tag_nums)): ...
            tags (Tensor(batch, seq_len)): ...
            mask (Tensor(batch, seq_len)): mask <bos> <eos > and <pad>
            crf: whether use crf to calculate loss
            transition: transition matrix
        '''

        if not use_crf:
            # (batch, seq_len, tag_nums) -> (sum_of_sentences_len, tag_nums)
            scores = scores[mask]
            # (batch, seq_len) -> sum_of_sentences_len
            target = tags[mask]
            loss = criterion(scores, target)
        else:
            loss = crf_forward(scores, tags, mask, transition)
            if torch.isnan(loss):
                raise Exception('loss is nan')
        return loss

    
    def decode(self, scores_arc, mask):
        pass

    @classmethod
    def build_tagger(cls, args):
        
        # directory to save model and fields
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        args.tagger_fields = os.path.join(args.save_dir, 'tagger_fields')
        args.tagger_model = os.path.join(args.save_dir, 'tagger_model')
        # TODO merge fields and model in a torch.save
        # build tagger fields
        if not os.path.exists(args.tagger_fields):
            print('Create fields for Tagger !')
            WORD = Field(pad_token=pad_token, unk_token=unk_token, bos_token=bos_token, 
                         eos_token=eos_token, lower=True)
            # TODO char-bilstm, use eos_token
            FEAT = SubWordField(pad_token=pad_token, unk_token=unk_token, bos_token=bos_token,
                                eos_token=eos_token, fix_len=args.fix_len, tokenize=list)
            # TODO need bos_token and eos_token?
            POS = Field(bos_token=bos_token, eos_token=eos_token)
            conll = Conll.load(args.ftrain)

            tri_fields = {
                'WORD': (WORD, Conll.field_names[1]),
                'FEAT': (FEAT, Conll.field_names[1]),
                'POS': (POS, Conll.field_names[3])
            }
            
            # field.build_vocab(getattr(conll, name), (Embedding.load(args.w2v, args.unk) if args.w2v else None))
            WORD.build_vocab(examples=getattr(conll, tri_fields['WORD'][1]), 
                             min_freq=args.min_freq, 
                             embed=(Embedding.load(args.w2v) if args.w2v else None))
            FEAT.build_vocab(examples=getattr(conll, tri_fields['FEAT'][1]))
            POS.build_vocab(examples=getattr(conll, tri_fields['POS'][1]))
        # TODO load fields
        else:
            pass
        # build tagger model
        # TODO
        tagger_model = Tagger_Model(n_words=WORD.vocab.n_init,
                                    n_chars=FEAT.vocab.n_init,
                                    n_tags=POS.vocab.n_init,
                                    n_embed=args.n_embed,
                                    n_char_embed=args.n_char_embed,
                                    n_feat_embed=args.n_feat_embed,
                                    n_lstm_hidden=args.n_lstm_hidden,
                                    n_lstm_layer=args.n_lstm_layer,
                                    pad_index=WORD.pad_index,
                                    unk_index=WORD.unk_index)

        # TODO save model
        if os.path.exists(args.tagger_model):
            state = torch.load(args.tagger_model, map_location=args.device)
            tagger_model.load_pretrained(state['pretrained'])
            tagger_model.load_state_dict(state['state_dict'], False)
        
        tagger_model.to(args.device)
        
        print('The network structure of POS Tagger is:\n', tagger_model)

        
        return cls(args, tri_fields, tagger_model)


        


                
            

            
            







