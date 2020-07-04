"""

	wmt16 en-ro data, without specific data size control

"""
import os
import sys
import pdb
import array
import struct
import numpy as np

# SAVE PATH
SAVE_PATH = ""

np.random.seed(42)


def save_data(fold, lang, data):
	with open(os.path.join(SAVE_PATH, "{}.tok.{}".format(fold, lang)), "w") as f:
		f.writelines(data)


def load_vocab():
	"""vocab, basically token2id"""
	vocab_path = os.path.join(SAVE_PATH, "vocab.bpe.32000")
	vocab = {}
	with open(vocab_path, "r") as f:
		for idx, line in enumerate(f.readlines()):
			token = line.strip()
			vocab[token] = idx
		vocab[''] = len(vocab)
	return vocab


def tensorize(sent, vocab):

	sent_enc = array.array('H')
	sent_enc.extend((vocab[token] for token in sent.split()))
	byte_rep = sent_enc.tobytes()
	byte_len = len(byte_rep)
	return struct.pack('Q{}s'.format(byte_len), byte_len, byte_rep)


def binarize(fold, vocab):
	"""
		
		binarize the bpe-ed file to .bin which will be read by the model
		preprocess steps: synst.src.data.annotated.preprocess_bpe

	"""
	tgt_path = os.path.join(SAVE_PATH, f'{fold}.tok.bpe.32000.ro')
	src_path = os.path.join(SAVE_PATH, f'{fold}.tok.bpe.32000.en')
	out_path = os.path.join(SAVE_PATH, f'{fold}.tok.bpe.32000.bin')
	with open(src_path, 'r') as f_src, \
			open(tgt_path, 'r') as f_tgt, \
			open(out_path, 'wb') as f_out:
		src_data = f_src.readlines()
		tgt_data = f_tgt.readlines()
		assert len(src_data) == len(tgt_data)

		for sent_s, sent_t in zip(src_data, tgt_data):
			if len(sent_s.strip()) == 0 or len(sent_t.strip()) == 0:
				continue
			sent_s = tensorize(sent_s, vocab)
			sent_t = tensorize(sent_t, vocab)
			f_out.write(sent_s)
			f_out.write(sent_t)

if __name__ == "__main__":

	SAVE_PATH = sys.argv[2]

	if sys.argv[1] == "binarize":
		vocab = load_vocab()
		binarize("train", vocab)
		binarize("test", vocab)
		binarize("valid", vocab)


