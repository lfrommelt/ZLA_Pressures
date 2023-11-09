import torch
import numpy as np



# todo: we need some architecture where sender and receiver are "connected" by gumbel-softmax. See Docstring for SRPolicyNet for details.
class GSPolicyNetLSTM(torch.nn.Module):
    #todo: implement eval mode/forward for inference
    # !!!the assholes from paper did actually frame the reconstruction problem as classification problem!!! Is zero-shot generalization due to compositionality even possible then???
        
    def __init__(self, in_shape, message_shape, out_shape, device="cpu", sender_hidden=250, receiver_hidden=100, classification=True):
        super().__init__()
        self.torch_device=device
        self.message_shape=message_shape#lets use agent as variable dump for a sec
        self.out_shape=out_shape
        self.in_shape = in_shape
        self.classification=classification
        #self.gs = lambda x: torch.nn.functional.gumbel_softmax(x, tau=1, hard=True, eps=1e-10, dim=-1)# can not be serialized that way...
        
        self.message = np.zeros(message_shape[0])
        
        self.sender_hidden=sender_hidden
        self.receiver_hidden=receiver_hidden
        
        self.sender1 = torch.nn.Linear(in_shape, self.sender_hidden)
        self.relu1 = torch.nn.ReLU()
        
        # freakin torch is a total shitshow!!! They give the option for a automatic projection for hidden states -> output, but then overwrite the hidden with this projection and not just the output!!?! So in the end we have to do the projection ourself, in order to access the hidden values. dafuq. stuff like this costs hours just to realize....
        self.rnn1 = torch.nn.LSTM(message_shape[1], self.sender_hidden, num_layers=1, proj_size=0)#message_shape[1])
        
        
        self.sos = torch.zeros(message_shape[1]) #not part of alphabet
        self.eos = torch.nn.functional.one_hot(torch.tensor(message_shape[1]-1)) # eos is part of alphabet
        self.c0 = torch.zeros(self.sender_hidden)
        
        self.projection = torch.nn.Linear(self.sender_hidden, message_shape[1])
        
        #linear layer sizes are completely arbitrary... (gotta be < receiver hidden in this case
        out_linear_hidden_size=self.receiver_hidden#int(receiver_hidden/2)
        self.h0=torch.zeros(out_linear_hidden_size)
        
        self.rnn2= torch.nn.LSTM(message_shape[1], self.receiver_hidden, num_layers=1)#, proj_size=out_linear_hidden_size)
        if classification:
            self.receiver2 = torch.nn.Linear(out_linear_hidden_size, out_shape[1]**out_shape[0])# here we could actually use torch.lstm projection instead
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        x1 = self.sender1(x)
        x2 = self.relu1(x1)

        '''
        torch gs doku: logits (Tensor) – […, num_features] unnormalized log probabilities
        wtf! logits do not have to be probs. unnormalized probs do not exist. log probs could be normalize in theory (never heard of it, though...)
        I guess the safest thing would be to use softmax(actual_logits aka x3).log() as input for gs. However using x3 as input did work (nans!)
        '''
        #.unsqueeze(0) for adding sequence dimension with size 1 (we have to do the recurrency part ourselves
        output, (hn, cn) = self.rnn1(self.sos.unsqueeze(0), (x2.unsqueeze(0), self.c0.unsqueeze(0)))
        output = self.projection(hn)
        
        #we need probs for aux losses, so we gotta do argmax trick our selves
        soft = torch.nn.functional.gumbel_softmax(output, tau=1, hard=False, eps=1e-10, dim=-1)
        #y_hard - y_soft.detach() + y_soft
        hard = torch.nn.functional.one_hot(soft.detach().argmax(dim=-1), num_classes=self.message_shape[1])
        symbol = hard - soft.detach() + soft
        nans=0
        '''
        There was a chance of around 1:1M - 2:1M for any testtensor that crashed training to produce a nan, probably because of sampling a low prob value.
        Solution: we simply resample. This introduces a bias (o
        '''
        while any(torch.isnan(symbol[0])):
            nans+=1
            soft = torch.nn.functional.gumbel_softmax(output, tau=1, hard=False, eps=1e-10, dim=-1)
            #y_hard - y_soft.detach() + y_soft
            hard = torch.nn.functional.one_hot(soft.detach().argmax(dim=-1), num_classes=self.message_shape[1])
            symbol = hard - soft.detach() + soft
            print(f"Warning: nan encountered!{nans}")
            if nans > 100:
                break#will lead to error, but rightly so
        
        symbols=[symbol[0]]
        soft_message=[soft]
        while not torch.all(symbol==self.eos):
            if len(symbols)==self.message_shape[0]-1:
                symbols.append(self.eos)#hm, this is a constant in case of len=max_len, but a differentiable tensor in other cases...
                break
            else:
                output, (hn, cn) = self.rnn1(symbols[-1].unsqueeze(0), (hn, cn))
                output = self.projection(hn)
                #we need probs for aux losses, so we gotta do argmax trick our selves
                soft = torch.nn.functional.gumbel_softmax(output, tau=1, hard=False, eps=1e-10, dim=-1)
                #y_hard - y_soft.detach() + y_soft
                hard = torch.nn.functional.one_hot(soft.detach().argmax(dim=-1), num_classes=self.message_shape[1])
                symbol = hard - soft.detach() + soft
                nans=0
                
                while any(torch.isnan(symbol[0])):
                    nans+=1
                    soft = torch.nn.functional.gumbel_softmax(output, tau=1, hard=False, eps=1e-10, dim=-1)
                    #y_hard - y_soft.detach() + y_soft
                    hard = torch.nn.functional.one_hot(soft.detach().argmax(dim=-1), num_classes=self.message_shape[1])
                    symbol = hard - soft.detach() + soft
                    print(f"Warning: nan encountered!{nans}")
                    if nans > 100:
                        break
                
                symbols.append(symbol[0])
                soft_message.append(soft)
        
        
        
        self.message_logits=torch.stack(soft_message)
        self.message = torch.stack(symbols)
        #print("message:",*self.message,sep="\n")
        
        #luckily for Listener we do not need to o the recursion ourselves
        _, (x5, _) = self.rnn2(self.message,(self.h0.unsqueeze(0), torch.zeros(self.receiver_hidden).unsqueeze(0)))
        x6 = self.receiver2(x5)#we use the batch dim as dim for n_attributes, if fails try with different weights per attribute
        return self.softmax(x6)
    
    def __repr__(self):
        return str(self.__class__)
    

# todo: we need some architecture where sender and receiver are "connected" by gumbel-softmax. See Docstring for SRPolicyNet for details.
class GSPolicyNet(torch.nn.Module):
    #todo: implement
        
    def __init__(self, in_shape, message_shape, out_shape, device="cpu", hidden1=100, hidden2=100):
        super().__init__()
        self.torch_device=device
        self.message_shape=message_shape#lets use agent as variable dump for a sec
        self.out_shape=out_shape
        self.sender1 = torch.nn.Linear(in_shape, hidden1)
        self.relu1 = torch.nn.ReLU()
        self.sender2 = torch.nn.Linear(hidden1, np.prod(message_shape))
        #self.gs = lambda x: torch.nn.functional.gumbel_softmax(x, tau=1, hard=True, eps=1e-10, dim=-1)# can not be serialized that way...
        self.receiver1 = torch.nn.Linear(np.prod(message_shape), hidden2)
        self.relu2 = torch.nn.ReLU()#could we recycle relu1?
        self.receiver2 = torch.nn.Linear(hidden2, np.prod(out_shape))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.message = np.zeros(message_shape[0])
    
    def forward(self, x):
        x1 = self.sender1(x)
        x2 = self.relu1(x1)
        x3 = self.sender2(x2)

        '''
        torch gs doku: logits (Tensor) – […, num_features] unnormalized log probabilities
        wtf! logits do not have to be probs. unnormalized probs do not exist. log probs could be normalize in theory (never heard of it, though...)
        I guess the safest thing would be to use softmax(actual_logits aka x3).log() as input for gs. However using x3 as input did work (nans!)
        '''
        soft = torch.nn.functional.gumbel_softmax(x3.view((-1,*self.message_shape)), tau=1, hard=False, eps=1e-10, dim=-1)
        #y_hard - y_soft.detach() + y_soft
        hard = torch.nn.functional.one_hot(soft.detach().argmax(dim=-1), num_classes=self.message_shape[1])
        message = hard - soft.detach() + soft  
        nans=0
        '''
        There was a chance of around 1:1M - 2:1M for any testtensor that crashed training to produce a nan, probably because of sampling a low prob value.
        Solution: we simply resample. This introduces a bias (o
        '''
        while any(torch.isnan(message.flatten())):
            nans+=1
            soft = torch.nn.functional.gumbel_softmax(x3.view((-1,*self.message_shape)), tau=1, hard=False, eps=1e-10, dim=-1)
            #y_hard - y_soft.detach() + y_soft
            hard = torch.nn.functional.one_hot(soft.detach().argmax(dim=-1), num_classes=self.message_shape[1])
            message = hard - soft.detach() + soft
            print(f"Warning: nan encountered!{nans}")
            if nans > 100:
                print("latents",x1,x2,x3,message,self.message,x4,x5,x6,x7,sep="\n")
                break#will lead to error, but rightly so
        self.message_logits=soft
        self.message = message#copy?
        x4 = self.receiver1(message.flatten(start_dim=-2))
        x5 = self.relu2(x4)
        x6 = self.receiver2(x5)
        x7 = self.softmax(x6.view(self.out_shape))#why even use RL at all? We could just use gs for the receiver output as well...

        return x7
    
    def __repr__(self):
        return str(self.__class__)
    
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