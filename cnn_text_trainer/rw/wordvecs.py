import cPickle
import os
import numpy as np

def load_wordvecs_from_binfile(word_vec_file,vocab=None):
    """
    Load word vectors from bin file
    :param word_vec_file: file path
    :param vocab: vocabulary. If not none, only words from this vocab will be loaded
    :return: dictionary of word to word_vector
    """
    with open(word_vec_file, "rb") as f:
        word_vecs = {}
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        i = 0
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if vocab == None or word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
        return word_vecs


def load_wordvecs(word_vec_file,vocab=None):
    i = 0
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    while not os.path.isfile(word_vec_file):       #TODO this is a hack. Find better way
        word_vec_file='../'+word_vec_file
        i+=1
        if i==4:
            raise Exception("File "+word_vec_file+" not found. Searched "+str(i)+" level above the cwd: till "+os.path.abspath(word_vec_file))

    word_vec_file = os.path.abspath(word_vec_file)

    os.chdir(cwd)
    if word_vec_file.endswith('.bin'):
        return load_wordvecs_from_binfile(word_vec_file,vocab)
    else:
        model=cPickle.load(open(word_vec_file,"rb"))
        word_idx_map, W = model[2], model[3]
        word_vecs = {}
        for word in word_idx_map:
            word_vecs[word]=W[word_idx_map[word]]
        return word_vecs


