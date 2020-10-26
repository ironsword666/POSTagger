python -u run.py \
    --device=4 \
    --seed=1 \
    --ftrain=data/ptb/train.conllx \
    --fdev=data/ptb/dev.conllx \
    --ftest=data/ptb/test.conllx \
    --w2v=data/embedding/glove.6B.100d.txt \
    --unk=unk \
    --save_dir=save/ \
    >log/pos.out 2>log/pos.err &
