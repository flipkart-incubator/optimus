import sys
from cnn_text_trainer.rw import datasets
from cnn_text_trainer.config import config
from cnn_text_trainer.core.multichannel.model import MultiChannelTrainer
from cnn_text_trainer.core.unichannel.model import TextCNNModelTrainer
from cnn_text_trainer.rw import wordvecs

__author__ = 'devashish.shankar'

if __name__=="__main__":
    if len(sys.argv)<5:
        print "Usage: training.py"
        print "\t<model config file path>"
        print "\t<training data file path>"
        print "\t<file path to store classifier model>"
        print "\t<true/false(preprocessing flag)>"
        exit(0)

    #processing..
    config_file=sys.argv[1]
    train_data_file=sys.argv[2]
    model_output_file=sys.argv[3]
    preprocess=sys.argv[4].lower()

    training_config = config.get_training_config_from_json(config_file)
    sentences, vocab, labels = datasets.build_data(train_data_file,preprocess)
    print "Dataset loaded"
    word_vecs = wordvecs.load_wordvecs(training_config.word2vec,vocab)
    print "Loaded word vecs from file"

    if training_config.mode=="multichannel":
        nntrainer = MultiChannelTrainer(training_config,word_vecs,sentences,labels)
    else:
        nntrainer = TextCNNModelTrainer(training_config,word_vecs,sentences,labels)

    nntrainer.train(model_output_file)



