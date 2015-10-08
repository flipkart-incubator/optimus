from collections import defaultdict
import csv
import re


def build_data(fname,preprocess=True):
    """
    Reads a CSV file with headers 'labels' and 'text' (containing label string and text respectively)
    and outputs sentences, vocab and labels
    :param fname: file name to read
    :param preprocess: should data be preprocessed
    :return: sentences is a list of dictionary (a format which NNTrainer accepts) [{'text': <text>, 'y':<label>},...]
    """
    sentences = []
    vocab = defaultdict(float)
    labels=[]
    rows = []
    with open(fname, "rb") as f:
        reader = csv.DictReader(f)
        count = 0
        for line in reader:
            if count%1000 == 0:
                print "Reading line no. ",count
            count+=1
            label = line['labels']
            rows.append((label,line['text']))  # Tuple: (label,text)
            labels=labels+[label]
    labels = list(set(labels))
    labels.sort()
    print labels
    labelIdToLabel = dict(zip(labels,range(len(labels))))
    for row in rows:
        y=labelIdToLabel[row[0]]
        rev = []
        rev.append(row[1].strip())
        if preprocess==True:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = rev[0]

        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1

        datum  = {"y":y,
                  "text": orig_rev,
                  "num_words": len(orig_rev.split())}
        sentences.append(datum)
    return sentences, vocab, labels


def clean_str(str):
    """
    Tokenization/string cleaning. This is specific to tweets, but can be applied for other texts as well.
    """
    str=str+" "
    str=re.sub("http[^ ]*[\\\]","\\\\",str)                    #Remove hyperlinks
    str=re.sub("http[^ ]* "," ",str)                           #Remove hyperlinks
    str=str.replace('\\n',' ')
    arr=re.findall(r"\w+(?:[-']\w+)*|'|[:)-.(]+|\S\w*", str)   #Single punctuation mark is removed, smileys remain intact
    arr=[i for i in arr if len(i)>1 and i[0]!='@']             #Remove words starting with @ (Twitter mentions)
    arr=[i if i[0]!='#' else i[1:] for i in arr]               #Remove '#' from hashtags
    #arr=[i for i in arr if i!='http' and i!='com' and i!='org']
    res=" ".join(arr)
    return res.lower().strip()
