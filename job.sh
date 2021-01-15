#!/bin/bash

# export CUDA_VISIBLE_DEVICES=

# python -u run.py \
#     --device=4 \
#     --seed=1 \
#     --feat=char \
#     --ftrain=data/ctb/train.conll \
#     --fdev=data/ctb/dev.conll \
#     --ftest=data/ctb/test.conll \
#     --w2v=data/embedding/giga_with_unk.100.txt \
#     --unk=UNK \
#     --save_dir=save/ctb/ \
#     >log/ctb5_local.out 2>log/ctb5_local.err &


# python -u run.py \
#     --device=3 \
#     --seed=1 \
#     --feat=char \
#     --use_crf \
#     --ftrain=data/ctb/train.conll \
#     --fdev=data/ctb/dev.conll \
#     --ftest=data/ctb/test.conll \
#     --w2v=data/embedding/giga_with_unk.100.txt \
#     --unk=UNK \
#     --save_dir=save/ \
#     >log/ctb5_crf_viterbi.out 2>log/ctb5_crf_viterbi.err &


python -u run.py \
    --device=3 \
    --seed=1 \
    --feat=char \
    --use_crf \
    --ftrain=data/ctb/train.conll \
    --fdev=data/ctb/dev.conll \
    --ftest=data/ctb/test.conll \
    --w2v=data/embedding/giga_with_unk.100.txt \
    --unk=UNK \
    --save_dir=save/ \
    >log/ctb5_crf_viterbi_test.out 2>log/ctb5_crf_viterbi_test.err &
