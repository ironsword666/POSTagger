# python -u run.py \
#     --device=5 \
#     --seed=1 \
#     --ftrain=data/ctb/train.conll \
#     --fdev=data/ctb/dev.conll \
#     --ftest=data/ctb/test.conll \
#     --w2v=data/embedding/giga_with_unk.100.txt \
#     --unk=UNK \
#     --save_dir=save/ctb/ \
#     >log/ctb5_local.out 2>log/ctb5_local.err &


python -u run.py \
    --device=3 \
    --seed=1 \
    --use_crf \
    --ftrain=data/ctb/train.conll \
    --fdev=data/ctb/dev.conll \
    --ftest=data/ctb/test.conll \
    --w2v=data/embedding/giga_with_unk.100.txt \
    --unk=UNK \
    --save_dir=save/ \
    >log/ctb5_crf.out 2>log/ctb5_crf.err &
