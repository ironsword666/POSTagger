from collections import Counter, OrderedDict
from vocab import Vocab


class RawField(object):
    ''' Defines a general datatype.

    Every dataset consists of one or more types of data. 
    For instance, a text classification dataset contains sentences and their classes, 
    while a machine translation dataset contains paired examples of text in two languages. 
    Each of these types of data is represented by a RawField object. 
    A RawField object does not assume any property of the data type and 
    it holds parameters relating to how a datatype should be processed.

    An example can be a sentence, a label sequence,  paired sentences ....
    '''

    def __init__(self, preprocessing=None, postprocessing=None):

        # self.name = name
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, data):
        ''' Preprocess an example if `preprocessing` is provided. '''
        if self.preprocessing:
            return self.preprocessing(data)
        else:
            return data

    def process(self, batch):
        ''' Process a list of examples to create a batch. 
        
        Params:
            batch (List[object]): a list of examples.
        '''

        if self.postprocessing:
            batch = self.postprocessing(batch)
        return batch


class Field(RawField):
    '''
    
    Attributes:
        tokenize: The function used to tokenize a string into token sequences. Default: ``None``.
        lower: Whether to lowercase the text.
        preprocessing:  The function that will be applied to examples using this field after tokenizing but before numericalizing.
    '''

    def __init__(self, pad_token=None, unk_token=None, bos_token=None, eos_token=None, 
                 fix_length=None, use_vocab=None, stop_word=None, 
                 tokenize=None, lower=False, preprocessing=None, postprocessing=None):

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.use_vocab = use_vocab
        self.stop_word = stop_word
        self.tokenize = tokenize
        self.lower = lower
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def build_vocab(self, dataset, specials=[]):
        ''' Construct the Vocab for this field from the source.

        Params:
            dataset (a list of examples): Used to get tokens.
            specials (list[str]): The list of special tokens.
        
        '''

        source = [getattr(dataset, name) for name, field in 
                  dataset.fields.items() if field is self]
        counter = Counter(token for data in source for token in self.preprocess(data))

        # use OrderedDict to keep tokens ordered and unique
        specials = list(OrderedDict.fromkeys(
            token for token in [self.pad, self.unk_token, self.bos_token,
                                self.eos_token] + specials
            if token is not None))

        self.vocab = Vocab(counter, specials)



    def preprocess(self, data):
        '''
        
        '''

        if self.tokenize:
            data = self.tokenize(data)
        if self.lower:
            data = [str(w) for w in data]
        if self.preprocessing:
            data = self.preprocessing(data)

        return data



    def process(self, batch):
        batch = self.numericalize(batch)
        batch = self.pad(batch)

        return batch

    def numericalize(self, batch):
        pass

    def pad(self, batch):
        pass

class SubWordField(Field):

    def __init__(self):
        pass