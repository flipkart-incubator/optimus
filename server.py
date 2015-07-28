import cPickle
import json
from flask import Flask, request
import sys
from cnn_text_trainer.rw.datasets import clean_str

__author__ = 'devashish.shankar'


#General refactoring, comments, etc.

app = Flask(__name__)
app.config['SECRET_KEY'] = 'F34TF$($e34D';   #Required for flask server TODO check

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
        models[model]=cPickle.load(open(model,"rb"))
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
        label_to_prob[labels[i]]=prob_pred[0][i]
    return json.dumps(label_to_prob)

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
    #In memory dictionary which will load all the models lazily
    models={}
    #In memory dictionary which will load all the word vectors lazily
    wordvecs={}

    #run app..

    app.run(debug=debug,host='0.0.0.0',port=port)
