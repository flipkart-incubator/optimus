
from collections import OrderedDict
import json
from flask import Flask, request
import sys
import numpy
from cnn_text_trainer.rw.datasets import clean_str

__author__ = 'devashish.shankar'


#General refactoring, comments, etc.

app = Flask(__name__)
app.config['SECRET_KEY'] = 'F34TF$($e34D';   #Required for flask server TODO check

import pickle

@app.route('/healthcheck')
def healthcheck():
    return json.dumps({})


@app.route('/')
def home():
    #The tweet to classify
    try:
        tweet=request.args['text'].lower()
    except Exception as e:
        print "Error processing request. Improper format of request.args['text'] might be causing an issue. Returning empty array"
        print "request.args['text'] = ",request.args['text']
        return json.dumps({})
    #The path to file containing the model
    model=str(request.args['model'])
    #Should the tweet be preprocessed
    preprocess=str(request.args['preprocess']).lower()
    #Lazily load the model
    if model not in models:
        print "Model not in memory: ",model
        print "Loading model"
        models[model]=pickle.load(open(model,"rb"))
        if(load_word_vecs):
            print "Adding wordvecs"
            models[model].add_global_word_vecs(wordvecs)
        print "Done"

    if preprocess == "True":
        tweet = clean_str(tweet)

    [y_pred,prob_pred] = models[model].classify([{'text':tweet}])
    labels = models[model].labels

    label_to_prob={}
    for i in range(len(labels)):
        if(isinstance(prob_pred[0][i], numpy.float32) or isinstance(prob_pred[0][i], numpy.float64)):
            label_to_prob[labels[i]]=prob_pred[0][i].item()
        else:
            label_to_prob[labels[i]] = prob_pred[0][i]
    return json.dumps(label_to_prob)


import logging

# Log only in production mode.
if not app.debug:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    app.logger.addHandler(stream_handler)

class LimitedSizeDict(OrderedDict):
  def __init__(self, *args, **kwds):
    self.size_limit = kwds.pop("size_limit", None)
    OrderedDict.__init__(self, *args, **kwds)
    self._check_size_limit()

  def __setitem__(self, key, value):
    OrderedDict.__setitem__(self, key, value)
    self._check_size_limit()

  def _check_size_limit(self):
    if self.size_limit is not None:
      while len(self) > self.size_limit:
        self.popitem(last=False)
#In memory dictionary which will load all the models lazily
models=LimitedSizeDict(size_limit=10)
#In memory dictionary which will load all the word vectors lazily
wordvecs={}

load_word_vecs = False

if __name__ == "__main__":
    if len(sys.argv)<4:
        print "Usage: server.py"
        print "\t<port number to deploy the app>"
        print "\t<enable flask debug mode (true/false). >"
        print "\t<load word vectors in memory (true/false). This will give accuracy gains, but will have a lot of memory pressure. If false, words not encountered during training are skipped while predicting >"
        exit(0)
    port=int(sys.argv[1])
    debug = sys.argv[2].lower()=="true"
    load_word_vecs = sys.argv[3].lower()=="true"

    #run app..

    app.run(debug=debug,host='0.0.0.0',port=port,threaded=True)
