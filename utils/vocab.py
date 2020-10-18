
class Vocab(object):

    def __init__(self, counter=None, min_freq=1, 
                 specials=['<pad>', '<unk>']):

        self.counter = counter

