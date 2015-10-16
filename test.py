import csv
import os
import sys
import cPickle
import operator
from cnn_text_trainer.rw import datasets

__author__ = 'devashish.shankar'

#TODO clean up this. Move to core maybe?
def evaluate(data,outputf):
    """
    Ported from initial version. TODO refactor to accept new format of data and clean up this code
    :param data: array containing outputs in old format: [[prob_pred,pred_label,actual_label,text],...]
    :param outputf: output directory
    """
    filept=open(outputf+"/info_"+testfile.split("/")[-1].split(".")[0]+"_"+modelfile.split("/")[-1].split(".")[0]+".csv", "wb")
    filep=csv.writer(filept)
    filep.writerow(["Number of data-points ",len(data)])
    print "Number of data-points: "+str(len(data))
    filep.writerow(["Number of labels ",len(labels)])
    print "Number of labels: "+str(len(labels))
    perf=float(len([row[1] for row in data if row[1]==row[2]]))/float(len(data))
    filep.writerow(["Accuracy ",str(perf*100)+"%"])
    filep.writerow([])
    print "Performance: "+str(perf*100)+"%\n"
    data.sort(key=operator.itemgetter(0),reverse=True)
    y_pred=[row[1] for row in data]
    y_true=[row[2] for row in data]
    for n in labels:
        tp=float(sum([(y_true[i]==n) and (y_pred[i]==n) for i in range(len(y_true))]))
        tn=float(sum([(y_true[i]!=n) and (y_pred[i]!=n) for i in range(len(y_true))]))
        fp=float(sum([(y_true[i]!=n) and (y_pred[i]==n) for i in range(len(y_true))]))
        fn=float(sum([(y_true[i]==n) and (y_pred[i]!=n) for i in range(len(y_true))]))
        fscore=(200*tp)/(2*tp+fp+fn)
        filep.writerow(["Label ",n])
        filep.writerow(["F-score  ",str(fscore)+"%"])
        filep.writerow(["TP ",int(tp),"FP ",int(fp),"TN ",int(tn),"FN ",int(fn)])
        filep.writerow([])
        print "F-score for label-"+str(n)+" is: "+str(fscore)+"%"
    filept.close()

    print "Printing output file"
    with open(outputf+"/output_"+testfile.split("/")[-1].split(".")[0]+"_"+modelfile.split("/")[-1].split(".")[0]+".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["probabilities","y_predicted","y_actual","tweets"])
        for line in data:
            writer.writerow(line)

    print "Printing misclassification file"
    with open(outputf+"/misclassification_"+testfile.split("/")[-1].split(".")[0]+"_"+modelfile.split("/")[-1].split(".")[0]+".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["probabilities","y_predicted","y_actual","tweets"])
        for line in data:
            if line[1]!=line[2]:
                writer.writerow(line)


if __name__=="__main__":
    if len(sys.argv)<6:
        print "Usage: testing.py"
        print "\t<model file path>"
        print "\t<testing file path>"
        print "\t<folder to store detailed output analysis>"
        print "\t<true/false preprocess>"
        print "\t<load word vectors? (true/false). This will give accuracy gains, but will have a lot of memory pressure. If false, words not encountered during training are skipped while testing >"
        exit(0)
    import theano
    theano.config.experimental.unpickle_gpu_on_cpu = True
    testfile=sys.argv[2]
    modelfile=sys.argv[1]
    outputdir=sys.argv[3]
    preprocess=sys.argv[4].lower()
    load_word_vecs = sys.argv[5].lower()=="true"

    if not os.path.exists(outputdir):
        print "Output dir ",outputdir, " doesn't exist. Creating it"
        os.makedirs(outputdir)
    else:
        print "Using Output dir ",outputdir,". Any previous results in this dir on same dataset might get overwritten. "
    model = cPickle.load(open(modelfile,"rb"))
    if load_word_vecs:
        print "Loading word vectors"
        model.add_global_word_vecs({})
        print "Loading word vectors done"
    sentences,vocab, labels = datasets.build_data(testfile,preprocess)
    labels = model.get_labels()
    output = model.classify(sentences)
    #Free memory
    del model
    print "Removed model from memory"
    #Format the output to earlier format
    #TODO evaluate function should be changed to accept newer format, which is cleaner
    data = []
    for i in range(len(output[0])):
        actual_label = sentences[i]['y']
        text = sentences[i]['text']
        predicted_label = output[0][i]
        predicted_prob = output[1][i][predicted_label]
        data.append([predicted_prob,labels[predicted_label],labels[actual_label],text])
    evaluate(data,outputdir)


