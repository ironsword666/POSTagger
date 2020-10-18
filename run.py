import argparse

from config import Config
from utils.preprocessor import read_file, create_vocab, read_vocab, load_pretrained_embedding
from utils.dataloader import TextDataSet, batchify
from learner.learner import train 

from mytagger.model import Model

'''
How to execute run.py?
使用crf
python run.py --word2vec_file=./data/embedding/glove.6B.100d.txt --unk=unk --use_crf
不使用crf
python run.py --word2vec_file=./data/embedding/glove.6B.100d.txt --unk=unk 
'''

if __name__ == '__main__':

    args = Config()

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--word2vec_file', default=None,
                        help='where to load pretrained embeddings')  
    parser.add_argument('--unk', default=None,
                        help='what is the label of unknown word in word2vec_file') 
    parser.add_argument('--use_crf', action='store_true',
                        help='whether use crf to calculate loss') 

    parser.parse_args(namespace=args)

    # get training set
    train_sentences = read_file(args.train_file)
    # drop out sentence whose length is larger than 'max_len'
    train_sentences = [sentence for sentence in train_sentences if len(sentence[0]) <= args.max_len]
    # build the vocabulary
    special_labels = [args.pad, args.unk, args.bos, args.eos]
    create_vocab(train_sentences, args.vocab_file, special_labels, 'word', args.min_freq)
    # words number in training set
    args.n_words = len(read_vocab(args.vocab_file))

    # load pretrained embeddings
    if args.word2vec_file:
        pretrained_embedding = load_pretrained_embedding(args.word2vec_file, args.n_embed, args.vocab_file, args.unk_pretrained)
    else:
        pretrained_embedding = None

    word_vocab = read_vocab(args.vocab_file)

    # create char vocab
    create_vocab(train_sentences, args.char_vocab_file, special_labels, 'char', args.min_freq)
    char_vocab = read_vocab(args.char_vocab_file)
    args.n_chars = len(char_vocab)
    # create tag vocab 
    # TODO should we add <UNK> to vocab ? since no <UNK> would 
    create_vocab(train_sentences, args.tag_vocab_file, special_labels, 'tag', min_freq=1)
    tag_vocab = read_vocab(args.tag_vocab_file)
    args.n_tags = len(tag_vocab)

    train_data = TextDataSet(args, train_sentences, word_vocab, char_vocab, tag_vocab)
    train_data_loader = batchify(train_data, args.batch_size, shuffle=True)
    print("create train_data_loader successfully !!!")

    dev_sentences = read_file(args.dev_file)
    dev_data = TextDataSet(args, dev_sentences, word_vocab, char_vocab, tag_vocab)
    dev_data_loader = batchify(dev_data, args.batch_size, shuffle=False)
    print("create dev_data_loader successfully !!!")

    test_sentences = read_file(args.test_file)
    test_data = TextDataSet(args, test_sentences, word_vocab, char_vocab, tag_vocab)
    test_data_loader = batchify(test_data, args.batch_size, shuffle=True)
    print("create test_data_loader successfully !!!")

    train(args, train_data_loader, dev_data_loader, test_data_loader, pretrained_embedding)


