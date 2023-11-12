from training.reinforce import MeanBaseline#maybe should go to utils
import torch
import numpy as np

class LengthLoss:
    def __init__(self, alpha):
        '''NOT the same as in paper. In paper speaker is trained with reinforce directly (and listener actually supervised!)'''
        self.baseline=MeanBaseline()
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
        probs=self.softmax(agent.message_logits)#.clone()
        
        #reward = self.alpha*((message==0)).sum()
        #print("message",message)
        
        #pressure = negative reward (kindof)
        pressure=len(message)-(message==0).sum()#len-n_zero is a distance: it is zero when everything is perfect
        #reward=self.baseline(reward)
        '''
        in the paper the sum of the task rewards and the alpha scaled length reward/loss are zero centered by the baseline
        since we do not use reinforce for sender update wrt task, the baseline would remove the scaling (cause we scale it to std,
        still not checked if that even makes sense/is the way to go), so we gotta aplly alpha afterwards. That way, task rewards are in the range [-1,1]
        
        '''
        pressure = self.alpha*pressure
        pressure = self.baseline(pressure)
        #print("pressure",pressure)
        
        rows = np.expand_dims([message!=0],-1)
        '''
        print(message)
        print(rows)
        print(message.reshape((len(message),1)))
        print(np.tile(np.arange(probs.shape[-1]),(len(message),1)))#==message.reshape((len(message),1)))# & rows)
        print(np.tile(np.arange(probs.shape[-1]),(len(message),1))==message.reshape((len(message),1)))'''
        mask=torch.from_numpy(np.tile(np.arange(probs.shape[-1]),(len(message),1))==message.reshape((len(message),1)).numpy())# & rows)#todo: checkout if constructing directly as tensor is faster (and howto)
        #print("relevant probs", torch.masked_select(probs,mask))
        #lets call it loss, since we minimize it
        #print("prob magnitude", torch.masked_select(probs, mask).sum())
        #print("log prob magnitude", torch.masked_select(torch.log(probs), mask).sum())
        
        #this can be minimized by either probs->0 aka 1/n_probs or len-n_zero -> 0
        loss=torch.prod(torch.masked_select(probs,mask))*pressure
        #print("loss",loss)
        #loss=-torch.sum(-torch.log((probs*reward)[((i,message[0][i]) for i in range(len(message)))]))
        #loss = -torch.sum([-torch.log(probs[i,message[i]]) * reward for i in range(len(message))])
        
        #raise Exception("try baseline")
        return loss
    
class LeastEffortLoss:
    '''
    We (implicitly) use CE(pred=message, gt=one_hot(message) as loss.
    The farmalization of the loss in the paper mainly focusses on doing softmax (we could use softmaxed logits, for now we test using
    the soft of the gumbel softmax, since this is the distribution that was actually used for sampling during training -> soft of gumbel
    leads to terrible stability, nans in weights after only one update
    '''
    def __init__(self, alpha=None):
        '''alpha just placeholder to make interfaces consistent'''
        self.baseline=MeanBaseline()
        self.softmax=torch.nn.Softmax(dim=-1)
        self.alpha=1

        
    def __repr__(self):
        return f"LeastEffort"
    
    def __call__(self, agent):
        '''
        getting agent makes the interface for different losses so much more easy. feels dirty tho. Only the most recent episode can be used/offline not possible
        note: the weight upates are not one wrt prob(value)*reward(message_i) but rather prob(value)*reward(message). This is one in order to be
        closer to the pressure from the paper, but could make learning harder
        '''
        message=agent.message.cpu().detach()#.argmax(dim=-1)[0]
        #probs=self.softmax(agent.message_logits)#.clone()
        probs=agent.message_probs
        
        loss=torch.sum(-torch.log(torch.masked_select(probs,message.bool())))

        return loss
    
    
    
class Dirichlet:
    """
    Class representing a Dirichlet Process
    Distribution over distributions: by sampling from this distribution, we also change it.
    Sampling is done by Speaker and then assumed to stem from the dirichlet process.
    """
    def __init__(self, alpha, alphabet_size):
        self.alpha=alpha

        #we assume that each symbol has already been used once at start
        self.occurances=np.ones(alphabet_size)
        self.n=alphabet_size
        
    def prob(self, symbol):
        '''
        params:
            symbol: int index of symbol
        '''
        prob=self.occurances[symbol]/(self.alpha+self.n-1)

        return prob
    
    def update(self, symbol):
        '''
        todo: vectorize
        '''
        self.occurances[symbol]+=1
        self.n+=1
        
    def __call__(self, symbol):
        '''
        I'm lazy. Symbol wise, not for complete message
        '''
        self.update(symbol)
        return self.prob(symbol)
        
        
class DirichletLoss:
    '''
    In paper the respective pressure is implemented via the reward r=sum(int(c_i==c_k)*log(p(c_k))). This reward is used in a supervised manner.
    How this is possible is beyond me (not connected to graph at all... maybe could become by using gumbel softmax and multiply result with logprobs).
    Therefore the reward is maximized, using a reinforce update.
    '''
    def __init__(self, alpha=2, differentiable=False, alphabet_size=40):
        '''alpha is dirichlet param'''
        self.alphabet_size=alphabet_size
        self.baseline=MeanBaseline()
        self.softmax=torch.nn.Softmax(dim=-1)
        self.alpha=alpha#should be >= 1, relates to alphabet size and to how many times dirichlet has already been used in total
        self.dirichlet=Dirichlet(self.alpha, alphabet_size)
        self.differentiable=differentiable
        
    def __repr__(self):
        return f"DirichletLoss(diff={str(self.differentiable)})"
    
    def __call__(self, agent):
        '''
        getting agent makes the interface for different losses so much more easy. feels dirty tho. Only the most recent episode can be used/offline not possible
        note: the weight upates are not one wrt prob(value)*reward(message_i) but rather prob(value)*reward(message). This is one in order to be
        closer to the pressure from the paper, but could make learning harder
        '''
        
        #.argmax(dim=-1)[0]
        message_probs=agent.message_probs#self.softmax(agent.message_logits)#.clone()
        
        if not self.differentiable:
            message=agent.message.cpu().detach()
            probs=[self.dirichlet(symbol) for symbol in message.argmax(dim=-1)[0]]#note that dirichlet changes during processing of the message, other way implemented but not tested yet

            reward=sum(probs)
        
        
            #reinforce update as loss
            loss=-torch.log(torch.prod(torch.masked_select(message_probs,message.bool())))*reward
            
        else:
            message=agent.message_with_grad
            
            for symbol in message.cpu().detach().argmax(dim=-1)[0]:
                self.dirichlet.update(symbol)
            
            #we multiply the one_hot encoded tensor with the symbol properties, expanded along the message dimension
            '''
            e.g.
            [[1,0,0]     [[0.3, 0.2, 0.5]    [[0.3, 0.0, 0.0]
             [0,0,1]] X   [0.3, 0.2, 0.5]] =  [0.0, 0.0, 0.5]]
             
            hopefully this results in differentiable indexing, probably will just produce nans in the back pass....
            '''
            reward=torch.sum(message*torch.tensor([self.dirichlet.prob(symbol) for symbol in range(self.alphabet_size)]).expand((agent.message_shape[0],-1)))
            loss=-reward
            
        return loss