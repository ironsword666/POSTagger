import sys
import os

from models.tagger_model import Tagger_Model
from utils.metric import Metric
from utils.algorithms import crf, viterbi
from utils.field import Field
from 

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

class Tagger(object):

    def __init__(self):

        pass

    def train(self, args):
        '''
        
        Params:
            - train_data_loader: 
        
        '''

        # pos model
        model = Tagger_Model(n_words=args.n_words,
                             n_chars=args.n_chars,
                             n_tags=args.n_tags,
                             n_embed=args.n_embed,
                             n_char_embed=args.n_char_embed,
                             n_feat_embed=args.n_feat_embed,
                             n_lstm_hidden=args.n_lstm_hidden,
                             n_lstm_layer=args.n_lstm_layer,
                             pad_index=args.pad_index,
                             unk_index=args.unk_index)  

        # load pretrained embeddings                
        model.load_pretrained(pretrained_embedding)
        print('The network structure of POS Tagger is:\n', model)

        # TODO select correct Loss function
        criterion = nn.CrossEntropyLoss() 
        # Adam Optimizer
        optimizer = Adam(model.parameters(), args.learning_rate) 
        # learning rate decrease
        # new_lr = initial_lr * gamma**epoch = initial_lr * 0.75**(epoch/5000)
        scheduler = ExponentialLR(optimizer, args.decay**(1/args.decay_steps)) 

        best_epoch, best_accuracy = 0, 0
        for epoch in range(1, args.epochs+1):
            print('training epoch {} :'.format(epoch))
            # training mode，dropout is useful
            model.train() 
            total_loss = 0 
            for words, chars, tags in train_data_loader:

                optimizer.zero_grad()
                # mask <bos> <eos > and <pad>
                mask = words.ne(args.pad_index) & words.ne(args.bos_index) & words.ne(args.eos_index)
                # compute score
                scores = model(words, chars) 
                if not args.use_crf:
                    criterion = nn.CrossEntropyLoss() 
                    loss = get_loss(criterion, scores, tags, mask, use_crf=False, transition=None)
                else:
                    # TODO is criterion right?
                    criterion = nn.NLLLoss()
                    loss = get_loss(criterion, scores, tags, mask, use_crf=True, transition=model.transition)
                # compute grad
                loss.backward() 
                # clip grad which is larger than args.clip
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                # backpropagation
                optimizer.step()
                # updat learning rate
                scheduler.step()
                total_loss += loss.item()
                break
            print('total_loss: {}'.format(total_loss))

            accuracy = evaluate(args, model, dev_data_loader)
            print('Accuracy: {}'.format(accuracy))
            if accuracy > best_accuracy:
                best_epoch = epoch

            if epoch - best_epoch > args.patience:
                break

    def evaluate(self, args, model, data_loader):
        ''''''
        
        model.eval()
        n_total, n_right = 0, 0 # 所有文本，分类对了的文本
        for words, chars, tags in data_loader:

            mask = words.ne(args.pad_index) & words.ne(args.bos_index) & words.ne(args.eos_index)
            scores = model(words, chars) 
            out = torch.argmax(scores, dim=-1)
            equal = torch.eq(out[mask], tags[mask])
            n_right += torch.sum(equal).item()
            n_total += torch.sum(mask)

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
            loss = crf(scores, tags, mask, transition)
        return loss

    def decode(self, scores_arc, mask):
        pass

    @classmethod
    def build_tagger(cls, args):
        
        # directory to save model and fields
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        args.fields = os.path.join(args.save_dir, 'tagger_fields')
        args.model = os.path.join(args.save_dir, 'tagger_model')
        
        if not os.path.exists(args.fields):
            print('Create fields for Tagger !')
            Field(pad_token=)






