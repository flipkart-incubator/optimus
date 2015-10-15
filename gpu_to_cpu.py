'''
Under development
'''
import theano
from theano.tensor.sharedvar import TensorSharedVariable



'''
Converts a CudaSharedVariable into a TensorSharedVariable.
This code should be run on a GPU machine.
'''
def cudaSharedVarToTensorSharedVar(cudaSharedVar):
    value = cudaSharedVar.get_value()
    theano4dtensor = theano.tensor.tensor4(dtype='float64')
    tensorType4d = theano4dtensor.__dict__['type']
    return TensorSharedVariable(name=cudaSharedVar['name'],type=tensorType4d,value=value,strict=False)