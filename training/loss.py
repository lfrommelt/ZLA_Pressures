from training.reinforce import Baseline#maybe should go to utils
import torch
import numpy as np

class LengthLoss:
    def __init__(self, alpha):
        '''NOT the same as in paper. In paper speaker is trained with reinforce directly (and listener actually supervised!)'''
        self.baseline=Baseline()
        self.alpha=alpha
        self.softmax=torch.nn.Softmax(dim=-1)

        
    def __repr__(self):
        return f"LengthLoss({self.alpha})"
    
    def __call__(self, agent):
        '''
        getting agent makes the interface for different losses so much more easy. feels dirty tho. Only the most recent episode can be used/offline not possible
        note: the weight upates are not one wrt prob(value)*reward(message_i) but rather prob(value)*reward(message). This is one in order to be
        closer to the pressure from the paper, but should make learning harder
        '''
        #"no-communication" symbol
        message=agent.message.cpu().detach().argmax(dim=-1)[0]
        probs=agent.message_logits#.clone()
        
        #reward = self.alpha*((message==0)).sum()
        #print("message",message)
        
        #pressure = negative reward (kindof)
        pressure=len(message)-(message==0).sum()
        #reward=self.baseline(reward)
        '''
        in the paper the sum of the task rewards and the alpha scaled length reward/loss are zero centered by the baseline
        since we do not use reinforce for sender update wrt task, the baseline would remove the scaling (cause we scale it to std,
        still not checked if that even makes sense/is the way to go), so we gotta aplly alpha afterwards. That way, task rewards are in the range [-1,1]
        
        '''
        pressure = self.alpha*pressure
        #print("pressure",pressure)
        
        rows = np.expand_dims([message!=0],1)
        print(message)
        print(rows)
        print(np.tile(np.arange(probs.shape[-1]),(len(message),1))==message.reshape((len(message),1)))# & rows)
        mask=torch.from_numpy(np.tile(np.arange(probs.shape[-1]),(len(message),1))==message.reshape((len(message),1)) & rows)#todo: checkout if constructing directly as tensor is faster (and howto)
        #print("relevant probs", torch.masked_select(probs,mask))
        1/0
        #lets call it loss, since we minimize it
        #print("prob magnitude", torch.masked_select(probs, mask).sum())
        #print("log prob magnitude", torch.masked_select(torch.log(probs), mask).sum())
        loss=torch.sum(torch.masked_select(probs,mask)*pressure)
        #print("loss",loss)
        #loss=-torch.sum(-torch.log((probs*reward)[((i,message[0][i]) for i in range(len(message)))]))
        #loss = -torch.sum([-torch.log(probs[i,message[i]]) * reward for i in range(len(message))])
        return loss