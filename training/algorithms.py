from timeit import time
import torch
import tqdm
#from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

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

class MeanBaseline:
    '''
    Zero centers the mean of the rewards in an iterative manner.
    '''
    def __init__(self):
        #note: assumes a reward of 0 before start -> negativ reward is meaningful from the beginning on, no zero divisions have to be caught
        self.n=1
        self.mean=0.0
        
    def __call__(self, reward):
        self.n+=1
        self.mean = (self.n-1)/self.n*self.mean+(1/self.n)*reward
        return reward-self.mean
    
class Baseline:
    '''
    TODO: look up if normalizing wrt variance is actually a thing (its not in anti-efficient coding but I have a vague memory of it in some baseline specific paper...
    Each sample will be zero centered and scaled to unit-std (is that a thing?, should it be var?) wrt. all previous samples.
    Implemented with optimized runtime, no need to enhance code
    '''
    def __init__(self):
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

    
def cross_entropy(prediction, target, dim=-1):
    '''
    what is wrong with torch??? why do have to do everything yourself if you don't want undocumented weird stuff to happen???
    just throw an error if inputs are not normalized, but don't just calculate stuff and call it cross entropy allthough it is not
    '''
    return -torch.sum(target*torch.log(prediction))
    

# main algorithms
class Algorithm(ABC):
    '''
    Abstract base class for any algorithm used in this repository.
    '''
    @abstractmethod
    def __init__(self, dataset, agent, trainset, lr=1e-3, device="cpu", baseline=MeanBaseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1, aux_losses=[], classification = True, lr_listen=0):
        pass
        
    @abstractmethod
    def step():
        pass
    
class SRSupervised(Algorithm):
    """
    Sender-Receiver-Supervised: Training signal for Listener is cross entropy. Speaker
    receives single-step reinforce updates with Listener's (detached) loss and the
    (log-) probabilities of the sampled tokens.
    """
    def __init__(self, dataset, agent, lr=1e-3, device="cpu", baseline=MeanBaseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1, aux_losses=[], classification = False, lr_listen=0):
        """
        Initialize the class instance with the given parameters.

        Parameters:
        -----------
        dataset : data.data.Dataloader
            An instance of the dataloader used for training.

        agent : YourAgentClass
            An instance of the agent class representing the learning algorithm.

        lr : float, optional
            The learning rate for the training process. Default is 1e-3.

        device : str, optional
            The device used for training. Default is "cpu".

        baseline : Baseline, optional
            The baseline used for reward computation. Default is an instance of MeanBaseline().

        logging : list, optional
            Log of previous training run, in case of continuing training.

        n_steps : int, optional
            The total number of training steps.

        verbosity : int, optional
            Verbosity level for debugging. Legacy, but could be usefull for debugging.

        eval_steps : int, optional
            Evaluation step for logging and checkpointing. Default is -1 (no evaluation).

        aux_losses : list, optional
            List of auxiliary losses to be applied during training.

        classification : bool, optional
            Flag indicating whether the task is a classification task, i.e. outputting
            the label of the target instead of reconstructing it.

        lr_listen : int, optional
            Flag to control learning rate adjustment. Default is 0 (no adjustment).

        Returns:
        --------
        None
        """
        self.agent=agent
        self.dataset=dataset
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
        self.speaker_optimizer=torch.optim.Adam(agent.speaker.parameters(), lr=self.lr)
        if lr_listen:
            self.listener_optimizer=torch.optim.Adam(agent.listener.parameters(), lr=lr_listen)
        else:
            self.listener_optimizer=torch.optim.Adam(agent.listener.parameters(), lr=self.lr)
        self.baseline=baseline
        #wtf drecks torch ce-loss does softmax itself?!?!?!?
        self.ce = cross_entropy
        self.aux_losses=aux_losses
        self.classification=classification
        #self.fixed = not "LSTM" in agent.__class__()
        
    def step(self, level_index):
        """
        Does a single update step. Implementation includes environment and everything.


        Parameters
        ----------
        level_index: int
            index for the level that is used wrt. the dataset
        """
        # time of individual steps will be logged for debugging
        at=time.time()
        
        level=self.dataset.dataset[level_index]
        #likelihoods of receiver actions
        if self.classification:
            action=self.agent(level)
            answer=np.random.choice(np.arange(len(self.dataset), dtype="int"), p=action.detach().cpu().numpy())
            # rewards are not actual rewards in Supervised, instead are used for logging task succes
            reward= 1.0 if answer==level_index else -1.0
        else:
            action=self.agent(level)
            answer=[np.random.choice(np.arange(self.n_values), p=probs) for probs in action.detach().cpu().numpy()]
            reward= 1.0 if all(answer==level.view((self.n_attributes, self.n_values)).argmax(dim=-1).numpy()) else -1.0


        self.agenttime+=time.time()-at
        lt=time.time()

        self.logging.append((reward+1)/2)
        
        del(reward)#in case I accidently use it in supervised
            
        if self.classification:
            listener_loss=self.ce(action, torch.nn.functional.one_hot(torch.tensor(level_index), len(self.dataset)).float())
        else:            listener_loss=self.ce(action,level.view((self.n_attributes,self.n_values)))

        '''
        #debugging stuff
        print("aux",aux_loss_values)
        print("task", task_loss)
        print("sum", torch.sum(torch.stack((task_loss, *aux_loss_values))),"\n")
        '''
        
        #get log probs of choosen action
        mask=torch.from_numpy(np.tile(np.arange(self.alphabet_size),(len(self.agent.message),1))==self.agent.message.reshape((-1,self.alphabet_size)).numpy())
        
        if self.aux_losses:
            aux_penalty=self.aux_losses[0].alpha*(self.message_length-sum(self.agent.message.cpu().detach().argmax(dim=-1)==0))
        else:
            aux_penalty=0
            
        speaker_penalty=self.baseline(listener_loss.detach()+aux_penalty)
        
        #log of joint probability. sum of log probs should be same, right?
        speaker_loss = torch.log(torch.prod(self.agent.speaker.message_probs[mask]))*speaker_penalty
        
        
        self.losstime+=time.time()-lt
        bt=time.time()
        
        #reset optimizer and calculate gradients/weight upates
        self.listener_optimizer.zero_grad()
        listener_loss.backward()
        
        self.speaker_optimizer.zero_grad()
        speaker_loss.backward()
        
        self.backtime+=time.time()-bt


        ut=time.time()
        #apply updates (no batching)
        self.listener_optimizer.step()
        self.speaker_optimizer.step()
        self.updatetime+=time.time()-ut
                
        self.n_steps+=1    
    
        
class Supervised(Algorithm):
    """
    Sender and receiver are connected by Gumbel-Softmax straight-through and therefore trained jointly wrt task. I.e. Receiver trains supervised with
    sum_attributes(Cross-entropy(yhat_attribute, y_attribute)) and y_attribute=input, yhat:=output of receiver.
    Auxilary losses however are trained with reinforce wrt the output distribution of Speaker. The output distribution is approximated by taking softmax
    of the Gumbel-Softmax logits (because torch-Gumbel-Softmax is kinda black-boxy...)
    """
    def __init__(self, dataset, agent, lr=1e-3, device="cpu", baseline=MeanBaseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1, aux_losses=[], classification = False, lr_listen=0):
        """
        Initialize the class instance with the given parameters.

        Parameters:
        -----------
        dataset : data.data.Dataloader
            An instance of the dataloader used for training.

        agent : YourAgentClass
            An instance of the agent class representing the learning algorithm.

        lr : float, optional
            The learning rate for the training process. Default is 1e-3.

        device : str, optional
            The device used for training. Default is "cpu".

        baseline : Baseline, optional
            The baseline used for reward computation. Default is an instance of MeanBaseline().

        logging : list, optional
            Log of previous training run, in case of continuing training.

        n_steps : int, optional
            The total number of training steps.

        verbosity : int, optional
            Verbosity level for debugging. Legacy, but could be usefull for debugging.

        eval_steps : int, optional
            Evaluation step for logging and checkpointing. Default is -1 (no evaluation).

        aux_losses : list, optional
            List of auxiliary losses to be applied during training.

        classification : bool, optional
            Flag indicating whether the task is a classification task, i.e. outputting
            the label of the target instead of reconstructing it.

        lr_listen : int, optional
            Flag to control learning rate adjustment. Default is 0 (no adjustment).

        Returns:
        --------
        None
        """
        self.agent=agent
        self.dataset=dataset
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
        #wtf drecks torch ce-loss does softmax itself?!?!?!?
        #self.ce=torch.nn.CrossEntropyLoss(reduction="sum")
        self.ce = cross_entropy
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
            action=self.agent(level)
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
        
        '''
        aux_loss_values=[]
        for aux_loss in self.aux_losses:
            aux_loss_values.append(aux_loss(self.agent))'''
            
        #aux_loss = torch.sum(torch.stack(aux_loss_values))
        if self.classification:
            listener_loss=self.ce(action, torch.nn.functional.one_hot(torch.tensor(level_index), len(self.dataset)).float())
        else:
            listener_loss=self.ce(action,level.view((self.n_attributes,self.n_values)))#sum([(-torch.log(action[i,answer[i]]) * reward) for i in range(len(answer)),*aux_loss_values])

        '''
        print("aux",aux_loss_values)
        print("task", task_loss)
        print("sum", torch.sum(torch.stack((task_loss, *aux_loss_values))),"\n")'''
        
        #get log probs of choosen action
        
        mask=torch.from_numpy(np.tile(np.arange(self.alphabet_size),(len(self.agent.message),1))==self.agent.message.reshape((-1,self.alphabet_size)).numpy())
        

        #print(listener_loss)
        #print(self.aux_losses[0].alpha)
        #print(self.message_length-sum(self.agent.message.cpu().detach().argmax(dim=-1)==0))
        
        if self.aux_losses:
            aux_loss=self.aux_losses[0](self.agent)
            
        
        self.losstime+=time.time()-lt
        bt=time.time()
        
        #reset optimizer and calculate gradients/weight upates
        self.optimizer.zero_grad()
        #listener_loss.backward(retain_graph=True)#hope this works as intended...
        
        if self.aux_losses:
            (listener_loss + aux_loss).backward()
            
        else:
            listener_loss.backward()
            
        self.backtime+=time.time()-bt


        ut=time.time()
        #apply updates (no batching)
        self.optimizer.step()
        self.updatetime+=time.time()-ut
                
        self.n_steps+=1    
    
    
class SRREINFORCE(Algorithm):
    """
    Sender-Receiver-REINFORCE: Both sender and receiver get the same reward(/penalty)
    for individual single-step reinforce updates.
    """
    def __init__(self, dataset, agent, lr=1e-3, device="cpu", baseline=MeanBaseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1, aux_losses=[], classification = False, lr_listen=0):
        """
        Initialize the class instance with the given parameters.

        Parameters:
        -----------
        dataset : data.data.Dataloader
            An instance of the dataloader used for training.

        agent : YourAgentClass
            An instance of the agent class representing the learning algorithm.

        lr : float, optional
            The learning rate for the training process. Default is 1e-3.

        device : str, optional
            The device used for training. Default is "cpu".

        baseline : Baseline, optional
            The baseline used for reward computation. Default is an instance of MeanBaseline().

        logging : list, optional
            Log of previous training run, in case of continuing training.

        n_steps : int, optional
            The total number of training steps.

        verbosity : int, optional
            Verbosity level for debugging. Legacy, but could be usefull for debugging.

        eval_steps : int, optional
            Evaluation step for logging and checkpointing. Default is -1 (no evaluation).

        aux_losses : list, optional
            List of auxiliary losses to be applied during training.

        classification : bool, optional
            Flag indicating whether the task is a classification task, i.e. outputting
            the label of the target instead of reconstructing it.

        lr_listen : int, optional
            Flag to control learning rate adjustment. Default is 0 (no adjustment).

        Returns:
        --------
        None
        """
        self.agent=agent
        self.dataset=dataset
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
        self.speaker_optimizer=torch.optim.Adam(agent.speaker.parameters(), lr=self.lr)
        if lr_listen:
            self.listener_optimizer=torch.optim.Adam(agent.listener.parameters(), lr=lr_listen)
        else:
            self.listener_optimizer=torch.optim.Adam(agent.listener.parameters(), lr=self.lr)
        self.baseline1=MeanBaseline()
        self.baseline2=MeanBaseline()
        #wtf drecks torch ce-loss does softmax itself?!?!?!?
        #self.ce=torch.nn.CrossEntropyLoss(reduction="sum")
        self.ce = cross_entropy
        self.aux_losses=aux_losses
        self.classification=classification
        #self.fixed = not "LSTM" in agent.__class__()
        
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
            action=self.agent(level)
            answer=np.random.choice(np.arange(len(self.dataset), dtype="int"), p=action.detach().cpu().numpy())
            # rewards are not actual rewards in Supervised, instead are used for logging task succes
            reward= 1.0 if answer==level_index else -1.0
        else:
            action=self.agent(level)
            answer=[np.random.choice(np.arange(self.n_values), p=probs) for probs in action.detach().cpu().numpy()]
            reward= 1.0 if all(answer==level.view((self.n_attributes, self.n_values)).argmax(dim=-1).numpy()) else -1.0

        self.agenttime+=time.time()-at
        lt=time.time()

        self.logging.append((reward+1)/2)
            
        if self.classification:
            listener_penalty=self.ce(action, torch.nn.functional.one_hot(torch.tensor(level_index), len(self.dataset)).float())
        else:
            listener_penalty=self.ce(action,level.view((self.n_attributes,self.n_values)))

        '''
        print("aux",aux_loss_values)
        print("task", task_loss)
        print("sum", torch.sum(torch.stack((task_loss, *aux_loss_values))),"\n")'''
        
        #get log probs of choosen action
        mask=torch.from_numpy(np.tile(np.arange(self.alphabet_size),(len(self.agent.message),1))==self.agent.message.reshape((-1,self.alphabet_size)).numpy())
        
        if self.aux_losses:
            aux_penalty=self.aux_losses[0].alpha*(self.message_length-sum(self.agent.message.cpu().detach().argmax(dim=-1)==0))
        else:
            aux_penalty=0
        
        speaker_penalty=self.baseline1(reward)
        speaker_loss=torch.sum((-torch.log(self.agent.speaker.message_probs[mask] * speaker_penalty)))

        listener_probs=action[range(len(action)), answer]
        
        
        listener_loss=sum([(-torch.log(action[i,answer[i]]) * speaker_penalty) for i in range(len(answer))])

        self.losstime+=time.time()-lt
        bt=time.time()
        
        #reset optimizer and calculate gradients/weight upates
        self.listener_optimizer.zero_grad()
        listener_loss.backward()
        
        self.speaker_optimizer.zero_grad()
        speaker_loss.backward()
        
        self.backtime+=time.time()-bt


        ut=time.time()
        #apply updates (no batching)
        self.listener_optimizer.step()
        self.speaker_optimizer.step()
        self.updatetime+=time.time()-ut
                
        self.n_steps+=1   
    
    
class REINFORCEGS:
    """
    REINFORCE-Gumbel-Softmax: Listener gets single-step reinforce updates. Speaker
    gets updates via a backpass through gumbel-softmax.
    """
    
    def __init__(self, dataset, agent, lr=1e-3, device="cpu", baseline=MeanBaseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1, aux_losses=[], classification = False, lr_listen=0):
        """
        Initialize the class instance with the given parameters.

        Parameters:
        -----------
        dataset : data.data.Dataloader
            An instance of the dataloader used for training.

        agent : YourAgentClass
            An instance of the agent class representing the learning algorithm.

        lr : float, optional
            The learning rate for the training process. Default is 1e-3.

        device : str, optional
            The device used for training. Default is "cpu".

        baseline : Baseline, optional
            The baseline used for reward computation. Default is an instance of MeanBaseline().

        logging : list, optional
            Log of previous training run, in case of continuing training.

        n_steps : int, optional
            The total number of training steps.

        verbosity : int, optional
            Verbosity level for debugging. Legacy, but could be usefull for debugging.

        eval_steps : int, optional
            Evaluation step for logging and checkpointing. Default is -1 (no evaluation).

        aux_losses : list, optional
            List of auxiliary losses to be applied during training.

        classification : bool, optional
            Flag indicating whether the task is a classification task, i.e. outputting
            the label of the target instead of reconstructing it.

        lr_listen : int, optional
            Flag to control learning rate adjustment. Default is 0 (no adjustment).

        Returns:
        --------
        None
        """
        self.agent=agent
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