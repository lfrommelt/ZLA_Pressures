from timeit import time
import torch
import tqdm
#from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt


#Utility Functions
def entropy(probs):
    '''wtf, pytorch should have an implentation for this...'''
    assert all(torch.isclose(torch.sum(probs),torch.tensor([1.0]))), probs
    assert not 0.0 in probs, probs
    return -torch.sum(probs*torch.log(probs))

def one_hot(values, n_values):
    one_hot_vector = np.eye(n_values)[values]
    return one_hot_vector#.astype("int")

#Utility Classes
class Baseline:
    '''
    TODO: look up if normalizing wrt variance is actually a thing (its not in anti-efficient coing but I have a vgue memory of it in some baseline specific paper...
    Each sample will be zero centered and scaled to unit-std (is that a thing?, should it be var?) wrt. all previous samples.
    Implemented with optimal runtime, no need to enhance code
    '''
    def __init__(self):
        #note: assumes a reward of 0 before start -> negativ reward is meaningful from the beginning on, no zero divisions have to be caught
        self.n=1
        self.mean=0.0
        self.s=0.0
        self.std=0.0
        
    def __call__(self, reward):
        self.n+=1
        mean = self.mean+(reward-self.mean)/self.n
        self.s = self.s+(reward-self.mean)*(reward-mean)
        self.mean=mean
        self.std=(self.s/(self.n))**(1/2)
        if not self.std==0:
            return (reward-self.mean)/self.std
        else:
            return reward-self.mean
        
class Supervised:
    """
    Sender and receiver are connected by Gumbel-Softmax straight-through and therefore trained jointly wrt task. I.e. Receiver trains supervised with
    sum_attributes(Cross-entropy(yhat_attribute, y_attribute)) and y_attribute=input, yhat:=output ist of receiver.
    Auxilary losses however are trained with reinforce wrt the output distribution of sender. The output distribution is approximated by taking softmax
    of the Gumbel-Softmax logits (because torch-Gumbel-Softmax is kinda black-boxy...
    """
    def __init__(self, dataset, agent, trainset, lr=1e-2, device="cpu", baseline=Baseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1, aux_losses=[], classification = True):
        """
        args
        ----------
            dataset : np.array(shape=(n_data, n_attributes), dtype=attribute_domain)
                data
            sender : SRPolicyNet
                sender
            receiver : SRPolicyNet
                receiver
            trainset : array
                mask for dataset
        kwargs
        ----------
            verbosity : int
                if < 0: no; else: each indicted epoch
            others only for continuing training, can be ignored/use default otherwise
        """
        self.agent=agent
        self.dataset=dataset
        self.trainset=trainset
        self.device=device
        self.agenttime=0
        self.losstime=0
        self.backtime=0
        self.updatetime=0
        self.n_steps=n_steps
        self.logging_steps=0
        self.alphabet_size=agent.message_shape[1]
        self.message_length=agent.message_shape[0]
        self.n_attributes=dataset.n_attributes
        self.n_values=dataset.n_values#we assume that the domains of all attributes have the same size
        self.verbosity=verbosity
        self.eval_steps=eval_steps
        self.logging=[]
        self.lr=lr
        self.optimizer=torch.optim.Adam(agent.parameters(), lr=self.lr)
        self.baseline=baseline
        self.ce=torch.nn.CrossEntropyLoss(reduction="sum")
        self.aux_losses=aux_losses
        self.classification=classification
        
    def step(self, level_index):
        """
        Does a single reinforce step. Implementation includes environment and everything.


        Parameters
        ----------
            n_attributes : int
                amount of attributes per datum
            n_values : int
                all attributes have the same domain, i.e. arange(n_values)
            distribution : str
                flag for the distribution type, kinda ugly but that way accesible as param of the constructor, also see respective methods
            distribution_param : float
                a parameter of the Zipf-Mandelbrot law
            data_size_scale : float
                see _create_dataset functions for details

        """
        # time of individual steps will be logged for debugging
        at=time.time()
        
        level=self.dataset.dataset[level_index]
        #likelihoods of receiver actions
        if self.classification:
            action=self.agent(level)[0]
            answer=np.random.choice(np.arange(len(self.dataset), dtype="int"), p=action.detach().cpu().numpy())
            # rewards are not actual rewards in Supervised, instead are used for logging task succes
            reward= 1.0 if answer==level_index else -1.0
        else:
            action=self.agent(level)
            answer=[np.random.choice(np.arange(self.n_values), p=probs) for probs in action.detach().cpu().numpy()]
            reward= 1.0 if all(answer==level.view((self.n_attributes, self.n_values)).argmax(dim=-1).numpy()) else -1.0
        #sample action
        #try:

        
        '''
        except ValueError as e:
            print("action:", action)
            print("datum:", level)
            print("state:", state)
            print(list(self.agent.named_parameters()))
            raise e'''

        self.agenttime+=time.time()-at
        lt=time.time()
        
        # rewards are not given per each correct one       

        #sum([1.0 if answer[i]==self.dataset[level][i] else -1.0/self.n_attributes for i in range(len(answer))])#why are python list operators not vectorized to begin with?

        self.logging.append((reward+1)/2)
        
        del(reward)#in case I accidently use it in supervised
        
        aux_loss_values=[]
        for aux_loss in self.aux_losses:
            aux_loss_values.append(aux_loss(self.agent))
            
        #aux_loss = torch.sum(torch.stack(aux_loss_values))
        if self.classification:
            task_loss=self.ce(action, torch.nn.functional.one_hot(torch.tensor(level_index), len(self.dataset)).float())
        else:
            task_loss=self.ce(action,level.view((self.n_attributes,self.n_values)))#sum([(-torch.log(action[i,answer[i]]) * reward) for i in range(len(answer)),*aux_loss_values])

        '''
        print("aux",aux_loss_values)
        print("task", task_loss)
        print("sum", torch.sum(torch.stack((task_loss, *aux_loss_values))),"\n")'''
        
        loss=torch.sum(torch.stack((task_loss, *aux_loss_values)))
        self.losstime+=time.time()-lt
        bt=time.time()
        
        #reset optimizer and calculate gradients/weight upates
        self.optimizer.zero_grad()
        aux_loss_values[0].backward()
        
        self.backtime+=time.time()-bt


        ut=time.time()
        #apply updates (no batching)
        self.optimizer.step()
        self.updatetime+=time.time()-ut
                
        self.n_steps+=1    
    
    
class REINFORCEGS:
    """
    Extremely specific REINFORCE implementation, so far entails environment and everything and requires gs-agents
    """
    
    def __init__(self, dataset, agent, trainset, lr=1e-2, device="cpu", baseline=Baseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1, aux_losses=[]):
        """
        args
        ----------
            dataset : np.array(shape=(n_data, n_attributes), dtype=attribute_domain)
                data
            sender : SRPolicyNet
                sender
            receiver : SRPolicyNet
                receiver
            trainset : array
                mask for dataset
        kwargs
        ----------
            verbosity : int
                if < 0: no; else: each indicted epoch
            others only for continuing training, can be ignored/use default otherwise
        """
        self.agent=agent
        self.trainset=trainset
        self.device=device
        self.agenttime=0
        self.losstime=0
        self.backtime=0
        self.updatetime=0
        self.n_steps=n_steps
        self.logging_steps=0
        self.alphabet_size=agent.message_shape[1]
        self.message_length=agent.message_shape[0]
        self.n_attributes=dataset.n_attributes
        self.n_values=dataset.n_values#we assume that the domains of all attributes have the same size
        self.verbosity=verbosity
        self.eval_steps=eval_steps
        self.logging=[]
        self.lr=lr
        self.optimizer=torch.optim.Adam(agent.parameters(), lr=self.lr)
        self.baseline=baseline
        
        
    def step(self, level):
        """
        Does a single reinforce step. Implementation includes environment and everything.


        Parameters
        ----------
            n_attributes : int
                amount of attributes per datum
            n_values : int
                all attributes have the same domain, i.e. arange(n_values)
            distribution : str
                flag for the distribution type, kinda ugly but that way accesible as param of the constructor, also see respective methods
            distribution_param : float
                a parameter of the Zipf-Mandelbrot law
            data_size_scale : float
                see _create_dataset functions for details

        """
        # time of individual steps will be logged for debugging
        at=time.time()
        
        #likelihoods of receiver actions
        action=self.agent(level)
        
        
        #sample action
        try:
            answer=[np.random.choice(np.arange(self.n_values), p=probs) for probs in action.detach().cpu().numpy()]
        
        except ValueError as e:
            print("action:", action)
            print("datum:", level)
            print("state:", state)
            print(list(self.agent.named_parameters()))
            raise e

        self.agenttime+=time.time()-at
        lt=time.time()
        
        # rewards are not given per each correct one        
        reward= sum([1.0 if x else -0.1 for x in np.equal(one_hot(answer,self.n_values).flatten(),level.detach().cpu().numpy())])
        #sum([1.0 if answer[i]==self.dataset[level][i] else -1.0/self.n_attributes for i in range(len(answer))])#why are python list operators not vectorized to begin with?

        self.logging.append((reward+len(answer)*-0.1)/(len(answer)*0.1+len(answer*1.0)))#rescale to [0.0, 1.9] so it can be use as sucess rate like metric
        
        #reward is averaged, baseline gets updated automatically when called
        reward=self.baseline(reward)
        
        aux_loss_values=[]
        for aux_loss in aux_losses:
            aux_loss_values.append(aux_loss(self.agent.message))

        task_loss=sum([*[(-torch.log(action[i,answer[i]]) * reward) for i in range(len(answer))],*aux_loss_values])


        self.losstime+=time.time()-lt
        bt=time.time()
        
        #reset optimizer and calculate gradients/weight upates
        self.optimizer.zero_grad()
        loss.backward()
        
        self.backtime+=time.time()-bt


        ut=time.time()
        #apply updates (no batching)
        self.optimizer.step()
        self.updatetime+=time.time()-ut
                
        self.n_steps+=1