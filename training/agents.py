import torch
import numpy as np

# todo: we need some architecture where sender and receiver are "connected" by gumbel-softmax. See Docstring for SRPolicyNet for details.
class GSPolicyNet(torch.nn.Module):
    #todo: implement
        
    def __init__(self, in_shape, message_shape, out_shape):
        super().__init__()
        self.message_shape=message_shape#lets use agent as variable dump for a sec
        self.out_shape=out_shape
        self.sender1 = torch.nn.Linear(in_shape, 20)
        self.relu1 = torch.nn.ReLU()
        self.sender2 = torch.nn.Linear(20, np.prod(message_shape))
        #self.gs = lambda x: torch.nn.functional.gumbel_softmax(x, tau=1, hard=True, eps=1e-10, dim=-1)# can not be serialized that way...
        self.receiver1 = torch.nn.Linear(np.prod(message_shape), 20)
        self.relu2 = torch.nn.ReLU()#could we recycle relu1?
        self.receiver2 = torch.nn.Linear(20, np.prod(out_shape))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.message = np.zeros(message_shape[0])
    
    def forward(self, x):
        x1 = self.sender1(x)
        x2 = self.relu1(x1)
        x3 = self.sender2(x2)
        #message = self.gs(x.view((-1,*self.message_shape)))
        message = torch.nn.functional.gumbel_softmax(x3.view((-1,*self.message_shape)), tau=1, hard=True, eps=1e-10, dim=-1)
        
        nans=0
        '''
        There was a chance of around 1:1M - 2:1M for any testtensor that crashed training to produce a nan, probably because of sampling a low prob value.
        Solution: we simply resample. This introduces a bias (o
        '''
        while any(torch.isnan(message.flatten())):
            nans+=1
            message = torch.nn.functional.gumbel_softmax(x3.view((-1,*self.message_shape)), tau=1, hard=True, eps=1e-10, dim=-1)
            print(f"Warning: nan encountered!{nans}")
            if nans > 100:
                print("latents",x1,x2,x3,message,self.message,x4,x5,x6,x7,sep="\n")
                break#will lead to error, but rightly so
        
        self.message = np.argmax(message.cpu().detach().numpy(), axis=-1)
        x4 = self.receiver1(message.flatten(start_dim=-2))
        x5 = self.relu2(x4)
        x6 = self.receiver2(x5)
        x7 = self.softmax(x6.view(self.out_shape))#why even use RL at all? We could just use gs for the receiver output as well...

        return x7
    
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