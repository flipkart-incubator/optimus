'''
Under development
'''
import theano
from theano.tensor.sharedvar import TensorSharedVariable
import pickle
import sys
from cnn_text_trainer.core.nn_classes import MLPDropout, UnpickledLayer
import os
theano.config.experimental.unpickle_gpu_on_cpu = True

'''
Converts a CudaSharedVariable into a TensorSharedVariable.
This code should be run on a GPU machine.
'''
def cudaSharedVarToTensorSharedVar4d(cudaSharedVar):
    value = cudaSharedVar.get_value()
    theano4dtensor = theano.tensor.tensor4(dtype='float64')
    tensorType4d = theano4dtensor.__dict__['type']
    return TensorSharedVariable(name=cudaSharedVar.__dict__['name'],type=tensorType4d,value=value,strict=False)

'''
Converts a CudaSharedVariable into a TensorSharedVariable.
This code should be run on a GPU machine.
'''
def cudaSharedVarToTensorSharedVarVector(cudaSharedVar):
    value = cudaSharedVar.get_value()
    theano4dtensor = theano.tensor.vector(dtype='float64')
    tensorType4d = theano4dtensor.__dict__['type']
    return TensorSharedVariable(name=cudaSharedVar.__dict__['name'],type=tensorType4d,value=value,strict=False)


'''
Converts a CudaSharedVariable into a TensorSharedVariable.
This code should be run on a GPU machine.
'''
def cudaSharedVarToTensorSharedVarMatrix(cudaSharedVar):
    value = cudaSharedVar.get_value()
    theano4dtensor = theano.tensor.matrix(dtype='float64')
    tensorType4d = theano4dtensor.__dict__['type']
    return TensorSharedVariable(name=cudaSharedVar.__dict__['name'],type=tensorType4d,value=value,strict=False)



'''
steps.
'''
def gpu_to_cpu(modelName):
    o = pickle.load(open(modelName))
    print('pickle.loaded')

    for i in range(len(o.conv_layers)):
        o.conv_layers[i].W = cudaSharedVarToTensorSharedVar4d(o.conv_layers[i].W)
        o.conv_layers[i].b = cudaSharedVarToTensorSharedVarVector(o.conv_layers[i].b)
    wbclassifier = []
    for i in range(len(o.classifier.layers)):
        w = cudaSharedVarToTensorSharedVarMatrix(o.classifier.layers[i].W.owner.inputs[0].owner.inputs[0])
        b = cudaSharedVarToTensorSharedVarVector(o.classifier.layers[i].b)
        wb = UnpickledLayer(w, b)
        wbclassifier.append(wb)
    o.classifier = MLPDropout(layers=wbclassifier, activations=o.classifier.activations)
    filename, file_extension = os.path.splitext(modelName)
    pickle.dump(o, open(filename+"_cpu.p", 'w'))
    print('pickle.dumped')

if __name__=='__main__':
    if len(sys.argv) < 2:
        print "too few arguements.\n usage: script <modelName>"
        exit()
    elif len(sys.argv) > 2:
        print "too many arguements.\n usage: script <modelName>"
        exit()

    gpu_to_cpu(sys.argv[1])

    # /var/lib/fk-ark-webservice/cnnClassifiers/cnnModel_1053650855.p