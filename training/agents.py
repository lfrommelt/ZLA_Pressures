import torch

# todo: we need some architecture where sender and receiver are "connected" by gumbel-softmax. See Docstring for SRPolicyNet for details.
class GSPolicyNet(torch.nn.Module):
    #todo: implement
    pass

class SRPolicyNet(torch.nn.Module):
    '''
    Sender-Receiver-Policy-Net
    So far this architecture works for both, sender and receiver, by using the receiver param. It could make sense to:
    A) Restructure so that the whole agent is one class, same interface as with gumbel-softmax
    B) Leave it like this, inconsistent interface between gs and this
    or
    C) Ignore because in the end we will probably use gs
    '''
    
    
    def __init__(self, in_shape, out_shape, device="cuda"):
        """init function, params do what their name suggests"""
        super().__init__()
        
        '''
        todo: maybe later...
        self.embeddings=[]
        
        for channel in range(in_shape):
            self.embedings.append(torch.nn.Embedding("hm, reducing space from 4 to smaller seems not so important...")'''
        self.linear1 = torch.nn.Linear(in_shape, 20)
        self.activation = torch.nn.ReLU()
        self.outlayers=[]
        for _ in range(out_shape[0]):
            linear2 = torch.nn.Linear(20, out_shape[1],device=device)
            softmax = torch.nn.Softmax()
            self.outlayers.append((linear2, softmax))

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        y_pred = []
        for layer in self.outlayers:
            y = layer[0](x)
            y_pred.append(layer[1](y))
        return torch.stack(y_pred)