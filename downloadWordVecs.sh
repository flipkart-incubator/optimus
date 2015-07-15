#!/bin/bash
if [ ! -f GoogleNews-vectors-negative300.bin ]; then
    echo "Downloading google word2vecs"
    perl ./make/gdown.pl "https://docs.google.com/uc?export=download&confirm=Kqnw&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM" GoogleNews-vectors-negative300.bin.gz
    echo "done downloading word2vec. Uncompressing them"
    gunzip  GoogleNews-vectors-negative300.bin.gz
    echo "done uncompressing word2vec"
fi

