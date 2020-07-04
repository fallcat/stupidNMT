#!/bin/bash

# required environmental variables" SUBWORD, WMT16_SCRIPTS, MOSES, RAW_PATH, DOWNLOAD_PATH

# BPE related
BPE_VOCAB=$PROCESS_PATH/vocab.bpe.freq.32000
BPE_CODES=$PROCESS_PATH/bpe.32000
MERGE_OPS=32000

# download files
python process_new_data/download_data.py en_ro $RAW_PATH

# data pathEN
cat $RAW_PATH/training-parallel-ep-v8/europarl-v8.ro-en.en $RAW_PATH/SETIMES.en-ro.en > $RAW_PATH/raw.en
cat $RAW_PATH/training-parallel-ep-v8/europarl-v8.ro-en.ro $RAW_PATH/SETIMES.en-ro.ro > $RAW_PATH/raw.ro
EN_TRAIN_RAW=$RAW_PATH/raw.en
RO_TRAIN_RAW=$RAW_PATH/raw.ro
EN_TRAIN_TOK=$PROCESS_PATH/train.tok.en
RO_TRAIN_TOK=$PROCESS_PATH/train.tok.ro
EN_TRAIN_BPE=$PROCESS_PATH/train.tok.bpe.32000.en
RO_TRAIN_BPE=$PROCESS_PATH/train.tok.bpe.32000.ro
EN_DEV_SGM=$RAW_PATH/dev/newsdev2016-roen-ref.en.sgm
RO_DEV_SGM=$RAW_PATH/dev/newsdev2016-enro-ref.ro.sgm
EN_TEST_SGM=$RAW_PATH/test/newstest2016-roen-ref.en.sgm
RO_TEST_SGM=$RAW_PATH/test/newstest2016-enro-ref.ro.sgm
EN_DEV_TOK=$PROCESS_PATH/valid.tok.en
RO_DEV_TOK=$PROCESS_PATH/valid.tok.ro
EN_TEST_TOK=$PROCESS_PATH/test.tok.en
RO_TEST_TOK=$PROCESS_PATH/test.tok.ro
EN_DEV_BPE=$PROCESS_PATH/valid.tok.bpe.32000.en
RO_DEV_BPE=$PROCESS_PATH/valid.tok.bpe.32000.ro
EN_TEST_BPE=$PROCESS_PATH/test.tok.bpe.32000.en
RO_TEST_BPE=$PROCESS_PATH/test.tok.bpe.32000.ro

# sennrich's script for preprocess romanian
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
N_THREADS=8

RO_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ro | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l ro -no-escape -threads $N_THREADS"
EN_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS"


# tokenize training set
echo "Tokenizing training set"
cat $RO_TRAIN_RAW | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ro | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l ro -no-escape -threads $N_THREADS > $RO_TRAIN_TOK
cat $EN_TRAIN_RAW | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $EN_TRAIN_TOK

# tokenize dev & test
echo "Tokenizing dev set"
$INPUT_FROM_SGM < $EN_DEV_SGM | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $EN_DEV_TOK
$INPUT_FROM_SGM < $RO_DEV_SGM | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ro | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l ro -no-escape -threads $N_THREADS > $RO_DEV_TOK
echo "Tokenizing test set"
$INPUT_FROM_SGM < $EN_TEST_SGM | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $EN_TEST_TOK
$INPUT_FROM_SGM < $RO_TEST_SGM | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ro | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l ro -no-escape -threads $N_THREADS > $RO_TEST_TOK

# Learn bpe codes from training set
echo "Learn bpe codes"
cat $EN_TRAIN_TOK $RO_TRAIN_TOK $EN_DEV_TOK $RO_DEV_TOK $EN_TEST_TOK $RO_TEST_TOK | $SUBWORD/learn_bpe.py -s $MERGE_OPS > $BPE_CODES

# apply bpe codes to training set
echo "apply bpe on training set"
$SUBWORD/apply_bpe.py -c $BPE_CODES < $EN_TRAIN_TOK > $EN_TRAIN_BPE
$SUBWORD/apply_bpe.py -c $BPE_CODES < $RO_TRAIN_TOK > $RO_TRAIN_BPE

echo "get vocab"
cat $EN_TRAIN_BPE $RO_TRAIN_BPE > $PROCESS_PATH/joint.tok.bpe.$MERGE_OPS
$SUBWORD/get_vocab.py --input $PROCESS_PATH/joint.tok.bpe.$MERGE_OPS --output $BPE_VOCAB

# encode text using the bpe codes, add vocab for reverse encoding
echo "encode using bpe codes"
for lang in en ro; do
	for f in $PROCESS_PATH/*.tok.$lang; do
		outfile=${f%.*}.bpe.$MERGE_OPS.$lang
		$SUBWORD/apply_bpe.py -c $BPE_CODES --vocabulary $BPE_VOCAB < $f > $outfile
		echo $outfile
	done
done

echo "get vocab"
cat $EN_TRAIN_BPE $RO_TRAIN_BPE $EN_DEV_BPE $RO_DEV_BPE $EN_TEST_BPE $RO_TEST_BPE > $PROCESS_PATH/joint.tok.bpe.$MERGE_OPS
$SUBWORD/get_vocab.py --input $PROCESS_PATH/joint.tok.bpe.$MERGE_OPS --output $BPE_VOCAB
cat $BPE_VOCAB | cut -f 1 -d ' ' > $PROCESS_PATH/vocab.bpe.$MERGE_OPS

# binarize bpe encoded data
echo $PROCESS_PATH
python process_new_data/process_enro.py binarize $PROCESS_PATH