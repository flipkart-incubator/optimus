"""
Methods for reading the config from file into a config object

TODO: type validation
"""
import json


class TrainingConfig:
    def __init__(self,dim=300,word2vec='GoogleNews-vectors-negative300.bin',mode='static',
                 max_l=56,filter_h=5,filter_hs=[3,4,5],conv_features=100,mlp_hidden_units=[],dropout_rate=0.5,
                 shuffle_batch=True,n_epochs=50,batch_size=50,lr_decay=0.95,conv_non_linear='relu',sqr_norm_lim=9):
        self.dim = dim
        self.word2vec = word2vec
        self.mode = check_training_mode(mode)
        self.max_l = max_l
        self.filter_h = filter_h
        self.filter_hs = filter_hs
        self.conv_features = conv_features
        if mlp_hidden_units is None:
            mlp_hidden_units = []
        self.mlp_hidden_units = mlp_hidden_units
        self.dropout_rate = dropout_rate
        self.shuffle_batch = shuffle_batch
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.conv_non_linear = conv_non_linear
        self.sqr_norm_lim = sqr_norm_lim


def check_training_mode(str):
    if not str in ['static','nonstatic','multichannel']:
        raise KeyError(str+' not a valid training mode')
    return str

def get_training_config_from_json(file):
    with open(file,'rb') as f:
        jsonConfigs = json.loads(f.read())
        return TrainingConfig(dim=jsonConfigs.get('dim',300),
                              word2vec=jsonConfigs.get('word2vec','GoogleNews-vectors-negative300.bin'),
                              mode=check_training_mode(jsonConfigs.get('mode','static')),
                              max_l=jsonConfigs.get('max_l',56),
                              filter_h=jsonConfigs.get('filter_h',5),
                              filter_hs=jsonConfigs.get('filter_hs',[3,4,5]),
                              conv_features=jsonConfigs.get('conv_features',100),
                              mlp_hidden_units=jsonConfigs.get('mlp_hidden_units',[50]),
                              dropout_rate=jsonConfigs.get('dropout_rate',0.5),
                              shuffle_batch=jsonConfigs.get('shuffle_batch',True),
                              n_epochs=jsonConfigs.get('n_epochs',50),
                              batch_size=jsonConfigs.get('batch_size',50),
                              lr_decay=jsonConfigs.get('lr_decay',0.95),
                              conv_non_linear=jsonConfigs.get('conv_non_linear','relu')
                             )