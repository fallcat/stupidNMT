#!/bin/bash

# pjpcess ja corpus
# 1. replace space with special token, 2. tokenize (characters separated by space)
# python pjpcess_enjp.py pjpcess_jp

# pjpcess en corpus
# get bpe codes and vocab

# joint vocab 

# binarize data

# download data
python process_new_data/download_data.py en_ja $RAW_PATH

# data path
#BASE_PATH=/mnt/nfs/work1/miyyer/simengsun/data/small_enjp/
EN_TRAIN_RAW=$RAW_PATH/train.raw.en
JA_TRAIN_RAW=$RAW_PATH/train.raw.ja
EN_TRAIN_TOK=$PROCESS_PATH/train.tok.en
JA_TRAIN_TOK=$PROCESS_PATH/train.tok.ja
JOINT_TRAIN_TOK=$PROCESS_PATH/train.tok.joint
EN_DEV_RAW=$RAW_PATH/dev.raw.en
JA_DEV_RAW=$RAW_PATH/dev.raw.ja
EN_TEST_RAW=$RAW_PATH/test.raw.en
JA_TEST_RAW=$RAW_PATH/test.raw.ja
EN_DEV_TOK=$PROCESS_PATH/valid.tok.en
JA_DEV_TOK=$PROCESS_PATH/valid.tok.ja
EN_TEST_TOK=$PROCESS_PATH/test.tok.en
JA_TEST_TOK=$PROCESS_PATH/test.tok.ja

# sennrich's script for prepjpcess jpmanian
#WMT16_SCRIPTS=/mnt/nfs/work1/miyyer/simengsun/other/wmt16-scripts
NORMALIZE_JAMANIAN=$WMT16_SCRIPTS/prepjpcess/normalise-jpmanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/prepjpcess/remove-diacritics.py
#MOSES=/mnt/nfs/work1/miyyer/simengsun/other/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FJAM_SGM=$MOSES/scripts/ems/support/input-fjpm-sgm.perl
N_THREADS=8

EN_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS"
JA_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ja | $REM_NON_PRINT_CHAR | $TOKENIZER -l ja -no-escape -threads $N_THREADS"

cat $EN_TRAIN_RAW | $EN_PREPROCESSING > $EN_TRAIN_TOK
cat $EN_DEV_RAW | $EN_PREPROCESSING > $EN_DEV_TOK
cat $EN_TEST_RAW | $EN_PREPROCESSING > $EN_TEST_TOK
cat $JA_TRAIN_RAW | $JA_PREPROCESSING > $JA_TRAIN_TOK
cat $JA_DEV_RAW | $JA_PREPROCESSING > $JA_DEV_TOK
cat $JA_TEST_RAW | $JA_PREPROCESSING > $JA_TEST_TOK

# process and binarize data

cat $EN_TRAIN_TOK $EN_DEV_TOK $EN_TEST_TOK $JA_TRAIN_TOK $JA_DEV_TOK $JA_TEST_TOK > $JOINT_TRAIN_TOK

python process_new_data/process_enja.py process_spm_vocab $PROCESS_PATH
python process_new_data/process_enja.py spm_bpe_encode $PROCESS_PATH
python process_new_data/process_enja.py binarize $PROCESS_PATH