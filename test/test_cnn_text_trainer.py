import cPickle
import os
from cnn_text_trainer.config.config import get_training_config_from_json
from cnn_text_trainer.core.unichannel.model import TextCNNModelTrainer
from cnn_text_trainer.rw import wordvecs
from cnn_text_trainer.rw.datasets import build_data

__author__ = 'devashish.shankar'

def test_config_reader():
    #TODO improve this test case, probably check if values are actually getting correctly parsed from config
    config  = get_training_config_from_json("testConfig.json")
    assert config.mode == "static"
    print config

def test_dataset_reader():
    sentences,vocabs,labels = build_data("../sample/datasets/sst_small_sample.csv")
    assert len(sentences) == 300
    assert len(labels) == 2
    assert "neg" in labels and "pos" in labels

def trainer_helper(configFile,dataSetFile,tempModel):
    print "Training model on ",configFile,dataSetFile
    config  = get_training_config_from_json(configFile)
    sentences, vocab, labels = build_data(dataSetFile,True)
    word_vecs = wordvecs.load_wordvecs(config.word2vec,vocab)
    trainer = TextCNNModelTrainer(config,word_vecs,sentences,labels)
    trainer.train(tempModel)
    print "Succesfully trained model on ",configFile,dataSetFile," and model is at ",tempModel
    print "Will proceed at testing the model on same data. If everything is correct, you should see the same accuracy"
    model = cPickle.load(open(tempModel,"rb"))
    op = model.classify(sentences)
    os.remove(tempModel)

def test_all_trainers():
    trainer_helper("../sample/configs/sampleMCConfig.json","../sample/datasets/sst_small_sample.csv","tempModel.p")
    trainer_helper("../sample/configs/sampleNonStaticConfig.json","../sample/datasets/sst_small_sample.csv","tempModel.p")
    trainer_helper("../sample/configs/sampleMCConfig.json","../sample/datasets/sst_small_sample.csv","tempModel.p")
    #TODO validate embeddings change in MC in test case
    #TODO validate if preprocess flag is working


