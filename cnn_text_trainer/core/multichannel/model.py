import cPickle

import numpy as np
import theano
import theano.tensor as T

from cnn_text_trainer.core.nn_classes import MLPDropout
from cnn_text_trainer.core.unichannel.model import TextCNNModelTrainer, sgd_updates_adadelta, TextCNNModel
from cnn_text_trainer.rw import wordvecs


class MultiChannelModel(object):

    def __init__(self,trainingConfig,conv_layers,classifier,word_idx_map,Words_static,Words_nonstatic,labels,img_h):
        self.trainingConfig = trainingConfig
        self.conv_layers = conv_layers
        self.classifier = classifier
        self.word_idx_map_static=word_idx_map.copy()
        self.word_idx_map_nonstatic=word_idx_map.copy()
        self.Words_static = Words_static
        self.Words_nonstatic = Words_nonstatic
        self.labels = labels
        self.img_h = img_h
        self.word_vecs = {}

    def get_labels(self):
        return self.labels

    def add_global_word_vecs(self,word_vec_dict):
        """
        This function should be called by the instantiator, this allows the model
        to pick up word vectors, if they pre exist in memory. If not, they are
        loaded from file

        :param word_vec_dict: Global word vector dictionary
        """
        if self.trainingConfig.word2vec in word_vec_dict:
            self.word_vecs = word_vec_dict[self.trainingConfig.word2vec]
        else:
            self.word_vecs = wordvecs.load_wordvecs(self.trainingConfig.word2vec)
            word_vec_dict[self.trainingConfig.word2vec] = self.word_vecs

    def _classify(self,dataset_static,dataset_nonstatic):
        """
        Classify method for static or non-static models.
        :param classifier: model
        :param conv_layers: list of convPoolLayer objects
        :param Words: Dictionary of word index to word vectors
        :param dataset: Indices of words for the current sentence/dataset
        :param dim: dimension of word vector
        :param img_h: length of sentence vector after padding
        :return: [y_pred,prob_pred] The probability for each class
        """
        x_static = T.imatrix('x_static')
        x_nonstatic = T.imatrix('x_nonstatic')
        y = T.ivector('y')
        Words_static = theano.shared(value = self.Words_static, name = "Words_static")
        Words_nonstatic = theano.shared(value = self.Words_nonstatic, name = "Words_nonstatic")

        test_pred_layers = []
        test_size = np.shape(dataset_static)[0]
        test_layer0_input_static = Words_static[T.cast(x_static.flatten(),dtype="int32")].reshape((test_size,1,self.img_h,self.Words_static.shape[1]))
        test_layer0_input_nonstatic = Words_nonstatic[T.cast(x_nonstatic.flatten(),dtype="int32")].reshape((test_size,1,self.img_h,self.Words_nonstatic.shape[1]))
        for i in range(len(self.conv_layers)/2):
            test_layer0_output = self.conv_layers[i].predict(test_layer0_input_nonstatic, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        for i in range(len(self.conv_layers)/2,len(self.conv_layers)):
            test_layer0_output = self.conv_layers[i].predict(test_layer0_input_static, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))

        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict(test_layer1_input)
        test_prob_pred = self.classifier.predict_p(test_layer1_input)
        test_model_all = theano.function([x_static,x_nonstatic], (test_y_pred,test_prob_pred))

        return test_model_all(dataset_static,dataset_nonstatic)

    def classify(self,sentences):
        datasets_static = []
        datasets_nonstatic = []
        for sentence in sentences:
            dataset_static,self.Words_static,self.word_idx_map_static = self.get_idx_from_sent(sentence["text"],self.Words_static,self.word_idx_map_static)
            dataset_nonstatic,self.Words_nonstatic,self.word_idx_map_nonstatic = self.get_idx_from_sent(sentence["text"],self.Words_nonstatic,self.word_idx_map_nonstatic)
            datasets_static.append(dataset_static)
            datasets_nonstatic.append(dataset_nonstatic)
        datasets_static=np.array(datasets_static,dtype="int32")
        datasets_nonstatic=np.array(datasets_nonstatic,dtype="int32")

        return self._classify(datasets_static,datasets_nonstatic)

    def get_idx_from_sent(self, sent,Words,word_idx_map):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        pad = self.trainingConfig.filter_h - 1
        for i in xrange(pad):
            x.append(0)
        words = sent.split()
        W=list(Words)

        for wd in words:
            if wd in word_idx_map:
                x.append(word_idx_map[wd])
            elif self.word_vecs!=None and wd in self.word_vecs:
                word_idx_map[wd]=len(word_idx_map)+1
                W.append(self.word_vecs[wd])
                x.append(word_idx_map[wd])

        max_l = self.trainingConfig.max_l
        while len(x) < max_l +2*pad:
            x.append(0)
        if len(x) > max_l +2*pad:
            x=x[:max_l +2*pad]
        Words=np.array(W,dtype=theano.config.floatX)

        return x,Words,word_idx_map





class MultiChannelTrainer(TextCNNModelTrainer):

    def save_model(self,outputPath):
        cPickle.dump(MultiChannelModel(self.trainingConfig,self.conv_layers,self.classifier,self.word_idx_map,self.Words[0].get_value(),self.Words[1].get_value(),self.labels,self.img_h), open(outputPath, "wb"))
        # cPickle.dump([self.classifier,self.conv_layers,self.word_idx_map,self.Words.get_value(),self.labels,self.trainingConfig.max_l,self.trainingConfig.filter_h,self.trainingConfig.dim,self.trainingConfig.mode,self.trainingConfig.word2vec], open(outputPath, "wb"))

    # def save_model(self,outputPath):
    #     cPickle.dump([self.classifier,self.conv_layers,self.word_idx_map,[self.Words[0].get_value(),self.Words[1].get_value()],self.labels,self.trainingConfig.max_l,self.filter_h,self.trainingConfig.dim,self.trainingConfig.mode,self.trainingConfig.word2vec], open(outputPath, "wb"))
    #

    def construct_models(self, x,y):
        """
        Get MLP and Conv Net objects. Also define the parameters to backpropogate into.
        :param x:
        :return:
        """
        rng = np.random.RandomState(3435)
        Words_nonstatic = theano.shared(value = self.U, name = "Words_nonstatic")
        Words_static = theano.shared(value = self.U, name = "Words_static")

        conv_layer_input_nonstatic = Words_nonstatic[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words_nonstatic.shape[1]))
        conv_layer_input_static = Words_static[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words_static.shape[1]))

        conv_layers = []
        conv_outputs = []
        for i in xrange(len(self.trainingConfig.filter_hs)):
            conv_layer, conv_output = self.construct_conv_layer(self.filter_shapes[i], self.pool_sizes[i], conv_layer_input_nonstatic, rng)
            conv_layer_s, conv_output_s = self.construct_conv_layer(self.filter_shapes[i], self.pool_sizes[i], conv_layer_input_static, rng)
            conv_layers.extend([conv_layer,conv_layer_s])
            conv_outputs.extend([conv_output,conv_output_s])
        conv_output = T.concatenate(conv_outputs, 1)
        mlp_input_size = self.trainingConfig.conv_features * len(conv_layers)
        classifier = MLPDropout(rng, input=conv_output, layer_sizes=[mlp_input_size]+self.trainingConfig.mlp_hidden_units+[len(self.labels)],
                                activations=self.hidden_layer_activations,
                                dropout_rates=[self.trainingConfig.dropout_rate])

        #define parameters of the model and update functions using adadelta
        params = classifier.params
        for conv_layer in conv_layers:
            params += conv_layer.params
        params += [Words_nonstatic]
        cost = classifier.negative_log_likelihood(y)
        dropout_cost = classifier.dropout_negative_log_likelihood(y)
        grad_updates = sgd_updates_adadelta(params, dropout_cost, self.trainingConfig.lr_decay, 1e-6, self.trainingConfig.sqr_norm_lim)

        self.Words = [Words_nonstatic,Words_static]
        return classifier, conv_layers,cost,grad_updates

