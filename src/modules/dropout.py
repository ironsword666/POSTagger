import torch
import torch.nn as nn

class Dropout(nn.Module):

    def __init__(self, p=0.5):
        '''
        Args:
            p (float): probabiltiy that neuron is dropped out.
        '''

        super().__init__()
        
        if p < 0 or p > 1:
            raise ValueError('dropout probability has to be between 0 and 1,',
                             'but get {}'.format(p))

        self.p = p

    def forward(self, x):

        return x

    def __repr__(self):

        return self.__class__.__name__ + '(' + 'p=' + str(self.p) + ')'


class InvertedDropout(Dropout):
    '''
    InvertedDropout which drop out any neuron with probability p.
    '''

    def forward(self, x):

        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask = binomial.sample(x.size())
            return mask * x / (1 - self.p)

        return x

class SharedDropout(Dropout):
    '''
    SharedDropout where words of a sentence share same dropout probability, 
    but different sentences don't share same dropout probability.

    >>> tensor([[[2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.]],
                [[2., 2., 0., 0., 2.],
                 [2., 2., 0., 0., 2.],
                 [2., 2., 0., 0., 2.]])
    '''

    def forward(self, x):
        
        if self.training:
            
            # (batch, hidden_size)
            x_i = x[:, 0]
            # use bernoulli to zero elements
            mask = x_i.new_empty(x_i.size()).bernoulli_(p=1-self.p)
            return mask.unsqueeze(1) * x / (1 -  self.p)

        return x

class IndependentDropout(Dropout):
    '''
    IndependentDropout which is used to drop out words as unknown word,
    but different features of a word have different dropout probability.

    If a word has N features (such as word_embed, char_embed), and M features are saved, 
    after drop out features, we should rescale features with coefficient N/M,
    which is analogous to dropout probability `p`.
    '''

    def forward(self, *items):
        '''
        Args:
            items (list(Tensor)): Tensor is of the size (batch, seq_len, dim)
        '''

        if self.training:
            # list(Tensor(batch, seq_len))
            masks = [x.new_empty(x.size()[:2]).bernoulli_(1-self.p) for x in items]
            # (batch, seq_len)
            total = sum(masks)
            # rescale coefficient of different words
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            return [item * mask.unsqueeze(-1) for item, mask in zip(items, masks)]
        
        return items