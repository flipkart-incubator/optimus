from collections import OrderedDict
import cPickle

import numpy as np

import theano
import theano.tensor as T
from cnn_text_trainer.core.nn_classes import LeNetConvPoolLayer, MLPDropout, Iden
from cnn_text_trainer.rw import wordvecs


class TextCNNModel(object):

    def __init__(self,trainingConfig,conv_layers,classifier,word_idx_map,Words,labels,img_h):
        self.trainingConfig = trainingConfig
        self.conv_layers = conv_layers
        self.classifier = classifier
        self.word_idx_map = word_idx_map
        self.Words = Words
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

    def _classify(self,dataset):
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
        x = T.imatrix('x')

        test_pred_layers = []
        test_size = np.shape(dataset)[0]
        Words = theano.shared(value = self.Words, name = "Words")
        test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,self.img_h,self.Words.shape[1]))
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))

        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict(test_layer1_input)
        test_prob_pred = self.classifier.predict_p(test_layer1_input)
        test_model_all = theano.function([x], (test_y_pred,test_prob_pred))

        return test_model_all(dataset)

    def classify(self,sentences):
        datasets = []
        for sentence in sentences:
            dataset = self.get_idx_from_sent(sentence["text"])
            datasets.append(dataset)
        datasets=np.array(datasets,dtype="int32")
        print "lds",len(datasets)
        print "dss",np.shape(datasets[0])
        return self._classify(datasets)

    def get_idx_from_sent(self, sent):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        pad = self.trainingConfig.filter_h - 1
        for i in xrange(pad):
            x.append(0)
        words = sent.split()
        W=list(self.Words)

        for wd in words:
            if wd in self.word_idx_map:
                x.append(self.word_idx_map[wd])
            elif self.word_vecs!=None and wd in self.word_vecs:
                self.word_idx_map[wd]=len(self.word_idx_map)+1
                W.append(self.word_vecs[wd])
                x.append(self.word_idx_map[wd])

        max_l = self.trainingConfig.max_l
        while len(x) < max_l +2*pad:
            x.append(0)
        if len(x) > max_l +2*pad:
            x=x[:max_l +2*pad]
        self.Words=np.array(W,dtype=theano.config.floatX)
        return x


class TextCNNModelTrainer(object):
    """
    Trainer class. Constructs the model, and trains the dataset.
    """

    def __init__(self,trainingConfig,word_vecs,sentences,labels):
        """
        Inititalize the trainer.
        trainingConfig = config object
        word_vecs = Dictionary of words to word vectors (as loaded from file)
        sentences = list of dictionary. Each sentence is represented by a dict and has two keys: text and y
        """
        self.trainingConfig = trainingConfig
        self.labels = labels
        self.hidden_layer_activations = [Iden]
        #Get index map and U
        self.U,self.word_idx_map = self.index_wordvecs(word_vecs)
        print "Converted word vectors to word_idx_map"

        #Convert training data into a matrix according to word vector indices (dataset)
        self.datasets,self.num_labels = self.get_dataset_from_sentences(sentences,
                                                             self.word_idx_map,
                                                             self.trainingConfig.max_l,
                                                             self.trainingConfig.filter_h,
                                                             self.trainingConfig.dim)
        print "Converted the dataset into matrix. Num Rows = ",len(self.datasets)," Num labels = ",self.num_labels
        self.filter_shapes,self.pool_sizes = self.init_convolution_layer_params()

        self.parameters = [("image shape",self.img_h,self.img_w),("filter shape",self.filter_shapes), ("hidden_units",trainingConfig.mlp_hidden_units),
                  ("dropout", trainingConfig.dropout_rate), ("batch_size",trainingConfig.batch_size),
                    ("learn_decay",trainingConfig.lr_decay), ("conv_non_linear", trainingConfig.conv_non_linear), ("mode", trainingConfig.mode)
                    ,("sqr_norm_lim",trainingConfig.sqr_norm_lim),("shuffle_batch",trainingConfig.shuffle_batch),("n_epochs",trainingConfig.n_epochs)]
        print "Parameters for model: ",self.parameters


    def train(self,modelOutputPath):
        train_model,test_model,val_model,n_train_batches,n_val_batches = self.construct_theano_functions()

        #Theano function for setting zero padding back to zero
        zero_vec_tensor = T.vector()
        if type(self.Words) is list:    #In case words have multiple channels, first is assumed to be non static
            set_zero = theano.function([zero_vec_tensor], updates=[(self.Words[0], T.set_subtensor(self.Words[0][0,:], zero_vec_tensor))])
        else:
            set_zero = theano.function([zero_vec_tensor], updates=[(self.Words, T.set_subtensor(self.Words[0,:], zero_vec_tensor))])
        zero_vec = np.zeros(self.img_w,dtype=theano.config.floatX)

        #start training over mini-batches
        epoch = 0
        best_val_perf = 0
        val_perf = 0
        test_perf = 0
        cost_epoch = 0
        while (epoch < self.trainingConfig.n_epochs):
            epoch = epoch + 1
            if self.trainingConfig.shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = train_model(minibatch_index)
                    set_zero(zero_vec)
            else:
                for minibatch_index in xrange(n_train_batches):
                    cost_epoch = train_model(minibatch_index)
                    set_zero(zero_vec)

            train_losses = [test_model(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_model(i) for i in xrange(n_val_batches)]
            val_perf = 1- np.mean(val_losses)
            print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf * 100., val_perf*100.))
            if val_perf >= best_val_perf:      #Only save the model if it's validation performace is better. This is to prevent overfitting
                best_val_perf = val_perf
                print "Saving the best model"
                self.save_model(modelOutputPath)
        print "Training finished. Best model is at ",modelOutputPath

    def construct_theano_functions(self):
        """
        Construct the theano functions for training, testing and validation
        :return:
        """
        #define model architecture
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')
        classifier, conv_layers,cost,grad_updates = self.construct_models(x,y)

        #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
        #extra data (at random)
        np.random.seed(3435)
        if self.datasets.shape[0] % self.trainingConfig.batch_size > 0:
            extra_data_num = self.trainingConfig.batch_size - self.datasets.shape[0] % self.trainingConfig.batch_size
            train_set = np.random.permutation(self.datasets)
            extra_data = train_set[:extra_data_num]
            new_data=np.append(self.datasets,extra_data,axis=0)
        else:
            new_data = self.datasets
        new_data = np.random.permutation(new_data)
        n_batches = new_data.shape[0]/self.trainingConfig.batch_size
        n_train_batches = int(np.round(n_batches*0.9))

        #divide train set into train/val sets
        train_set = new_data[:n_train_batches*self.trainingConfig.batch_size,:]
        val_set = new_data[n_train_batches*self.trainingConfig.batch_size:,:]
        train_set_x, train_set_y = shared_dataset((train_set[:,:self.img_h],train_set[:,-1]))
        val_set_x, val_set_y = shared_dataset((val_set[:,:self.img_h],val_set[:,-1]))
        n_val_batches = n_batches - n_train_batches
        val_model = theano.function([index], classifier.errors(y),
             givens={
                x: val_set_x[index * self.trainingConfig.batch_size: (index + 1) * self.trainingConfig.batch_size],
                y: val_set_y[index * self.trainingConfig.batch_size: (index + 1) * self.trainingConfig.batch_size]})

        #compile theano functions to get train/val/test errors
        test_model = theano.function([index], classifier.errors(y),
                 givens={
                    x: train_set_x[index * self.trainingConfig.batch_size: (index + 1) * self.trainingConfig.batch_size],
                    y: train_set_y[index * self.trainingConfig.batch_size: (index + 1) * self.trainingConfig.batch_size]})
        train_model = theano.function([index], cost, updates=grad_updates,
              givens={
                x: train_set_x[index*self.trainingConfig.batch_size:(index+1)*self.trainingConfig.batch_size],
                y: train_set_y[index*self.trainingConfig.batch_size:(index+1)*self.trainingConfig.batch_size]})

        self.classifier = classifier
        self.conv_layers = conv_layers
        return train_model,test_model,val_model,n_train_batches,n_val_batches

    def construct_conv_layer(self, filter_shape, pool_size, layer0_input, rng):
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                        image_shape=(self.trainingConfig.batch_size, 1, self.img_h, self.img_w),
                                        filter_shape=filter_shape, poolsize=pool_size,
                                        non_linear=self.trainingConfig.conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        return conv_layer, layer1_input

    def construct_models(self, x,y):
        """
        Get MLP and Conv Net objects. Also define the parameters to backpropogate into.
        :param x:
        :return:
        """
        rng = np.random.RandomState(3435)
        Words = theano.shared(value = self.U, name = "Words")
        conv_layer_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
        conv_layers = []
        conv_outputs = []
        for i in xrange(len(self.trainingConfig.filter_hs)):
            conv_layer, conv_output = self.construct_conv_layer(self.filter_shapes[i], self.pool_sizes[i], conv_layer_input, rng)
            conv_layers.append(conv_layer)
            conv_outputs.append(conv_output)
        conv_output = T.concatenate(conv_outputs, 1)
        mlp_input_size = self.trainingConfig.conv_features * len(conv_layers)
        classifier = MLPDropout(rng, input=conv_output, layer_sizes=[mlp_input_size]+self.trainingConfig.mlp_hidden_units+[len(self.labels)],
                                activations=self.hidden_layer_activations,
                                dropout_rates=[self.trainingConfig.dropout_rate])

        #define parameters of the model and update functions using adadelta
        params = classifier.params
        for conv_layer in conv_layers:
            params += conv_layer.params
        if self.trainingConfig.mode=="nonstatic":
            #if word vectors are allowed to change, add them as model parameters
            params += [Words]
        cost = classifier.negative_log_likelihood(y)
        dropout_cost = classifier.dropout_negative_log_likelihood(y)
        grad_updates = sgd_updates_adadelta(params, dropout_cost, self.trainingConfig.lr_decay, 1e-6, self.trainingConfig.sqr_norm_lim)

        self.Words = Words
        return classifier, conv_layers,cost,grad_updates

    def init_convolution_layer_params(self):
        """
        Initialize configs for conv layers and max pooling
        filter_shapes: list of [shape of convolution filter]
        pool_sizes: list of pool size
        """
        self.img_w = self.trainingConfig.dim
        self.img_h = len(self.datasets[0])-1
        filter_w = self.img_w
        conv_features = self.trainingConfig.conv_features
        filter_shapes = []
        pool_sizes = []
        for filter_h in self.trainingConfig.filter_hs:
            filter_shapes.append((conv_features, 1, filter_h, filter_w))
            pool_sizes.append((self.img_h-filter_h+1, self.img_w-filter_w+1))
        return filter_shapes,pool_sizes

    def save_model(self,outputPath):
        cPickle.dump(TextCNNModel(self.trainingConfig,self.conv_layers,self.classifier,self.word_idx_map,self.Words.get_value(),self.labels,self.img_h), open(outputPath, "wb"))
        # cPickle.dump([self.classifier,self.conv_layers,self.word_idx_map,self.Words.get_value(),self.labels,self.trainingConfig.max_l,self.trainingConfig.filter_h,self.trainingConfig.dim,self.trainingConfig.mode,self.trainingConfig.word2vec], open(outputPath, "wb"))

    def index_wordvecs(self,word_vecs):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        word_vecs = Dictionary of words to word vectors (as loaded from file)
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, self.trainingConfig.dim),dtype=theano.config.floatX)
        W[0] = np.zeros(self.trainingConfig.dim,dtype=theano.config.floatX)
        i = 1
        for word in word_vecs:     #Iterate over keys
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def get_dataset_from_sentences(self,sentences, word_idx_map, max_l, filter_h, k):
        """
        Transforms sentences into a 2-d matrix.
        """
        train = []
        num_labels=0
        for sentence in sentences:
            sent = self.get_idx_from_sent(sentence["text"], word_idx_map, max_l, filter_h, k)
            sent.append(sentence["y"])
            if(sentence["y"]>num_labels):
                num_labels=sentence["y"]
            train.append(sent)
        train = np.array(train,dtype="int")
        return train,num_labels+1

    def get_idx_from_sent(self,sent, word_idx_map, max_l, filter_h, k):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        pad = filter_h - 1
        for i in xrange(pad):
            x.append(0)
        words = sent.split()
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
        while len(x) < max_l+2*pad:
            x.append(0)
        if len(x) > max_l+2*pad:
            x=x[:max_l+2*pad]
        return x

#Helper methods for the NN
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to

