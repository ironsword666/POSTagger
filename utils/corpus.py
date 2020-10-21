# -*- coding: utf-8 -*-

class Corpus(object):
    ''' Defines a general datatype.

    An example can be a sentence, a label sequence,  paired sentences ....
    '''

    def __init__(self):
        pass

    @classmethod
    def load(cls, path):
        return cls() 

    def save(self, path):
        pass

# TODO
class Sentence(object):

    def __init__(self):
        pass



class Conll(Corpus):

    fields = ['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __getattr__(self, name):
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for s in self.sentences:
            yield getattr(s, name)
        
    # TODO __setattr__()

    @classmethod
    def load(cls, path):

        with open(path, 'r', encoding='UTF-8') as fr:
            lines = [line.strip() for line in fr]

        start, sentences = 0, []
        for i, line in enumerate(lines):
            if not line: 
                # [[id, form, ...], [id, form, ...], ...]
                sentence = [line.split('\t') for line in lines[start:i]]
                # [(1, 2, 3, ...), (In, an, Oct, ...), ...]
                values = list(zip(*sentence))
                sentences.append(ConllSentence(Conll.fields, values))
                start = i + 1
        
        return cls(sentences)

    def save(self, path):
        pass

class ConllSentence(Sentence):

    def __init__(self, fields, values):
        for name, value in zip(fields, values):
            setattr(self, name, value)
        self.fields = fields
        self.length = len(getattr(self, fields[0]))

    def __len__(self):
        return self.length

    # TODO
    def __repr__(self):

        return None

# TODO
class Embedding(Corpus):

    def __init__(self):

        pass

    @classmethod
    def load(cls, path):

        pass