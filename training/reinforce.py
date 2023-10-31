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
    '''should maybe happen in Dataset so we don't have to do it again and again'''
    one_hot_vector = np.eye(n_values)[values]
    return one_hot_vector#.astype("int")

#Utility Classes
class Baseline:
    '''
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
        return (reward-self.mean)/self.std

class REINFORCE:
    """
    Extremely specific REINFORCE implementation, so far entails environment and everything...
    NOT up to date
    """
    
    def __init__(self, dataset, sender_policy, receiver_policy, trainset, lr=1e-2, device="cpu", baseline=Baseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1):
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
        self.dataset=dataset
        self.sender_policy=sender_policy
        self.receiver_policy=receiver_policy
        self.trainset=trainset
        self.device=device
        self.sendertime=0
        self.listenertime=0
        self.losstime=0
        self.backtime=0
        self.updatetime=0
        self.n_steps=n_steps
        self.logging_steps=0
        self.alphabet_size=sender_policy.outlayers[0][0].out_features
        self.message_length=len(sender_policy.outlayers)
        self.n_attributes=len(dataset[0])
        self.n_values=dataset.max()+1#we assume that the domains of all attributes have the same size
        self.n_distractors=int((receiver_policy.linear1.in_features-(self.alphabet_size*self.message_length))/(self.n_attributes*self.n_values))-1
        self.verbosity=-1
        self.eval_steps=eval_steps
        self.logging=[]
        self.sendertime=0
        self.listenertime=0
        self.losstime=0
        self.backtime=0
        self.updatetime=0
        self.lr=lr
        self.sender_optimizer = torch.optim.Adam(sender_policy.parameters(), lr=self.lr)
        self.receiver_optimizer = torch.optim.Adam(receiver_policy.parameters(), lr=self.lr)
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
        st=time.time()

        #randomly sample a distractor from trinset, that is actually different from the target
        distractors=np.zeros((self.n_distractors+1,self.n_attributes),dtype="int")# rename distractors <- stimuli
        distractors[0]=self.dataset[level]
        j=0
        while j<self.n_distractors:
            candidate=np.random.choice(self.trainset)
            if not all(self.dataset[candidate] == self.dataset[level]):
                j+=1
                distractors[j]=self.dataset[candidate]
        #transform datum into state for receiver (could be done beforehand for efficiency reasons)
        s_sender=torch.from_numpy(one_hot(self.dataset[level],self.n_values).flatten())
        s_sender=s_sender.type(torch.float).to(self.device)
        
        #genareate probability distribution over sender actions...
        a_sender=self.sender_policy(s_sender)
        probs = [prob.detach().cpu().numpy() for prob in a_sender]
        #... and sample the action from it
        message=[np.random.choice(np.arange(self.alphabet_size), p=prob) for prob in probs]

        self.sendertime+=time.time()-st

        lt=time.time()
        
        #transform message back so it fits its part of the receivers state space (aka one-hot encode)
        s_receiver=np.array([[1.0 if i==symbol else 0 for i in range(self.alphabet_size)] for symbol in message])
        
        #indexing array "shuffle", to shuffle the order of target and distractor(s)
        shuffle=np.arange(self.n_distractors+1,dtype="int")
        np.random.shuffle(shuffle)
        
        #actual statespace of receiver, i.e. conctenate(*shuffled(Target,*Distractors), message)
        s_receiver=torch.from_numpy(np.concatenate((*[one_hot(dist,self.n_values).flatten() for dist in distractors[shuffle]],s_receiver.flatten()))).type(torch.float).to(self.device)
        #likelihoods of receiver actions
        a_receiver=self.receiver_policy(s_receiver)[0]
        
        #sample from it
        answer=np.random.choice(np.arange(self.n_distractors+1), p=a_receiver.detach().cpu().numpy())

        self.listenertime+=time.time()-lt
        lt=time.time()
        
        reward= 1.0 if shuffle[answer]==0 else -1.0

        self.logging.append(reward)
        
        #reward is averaged, baseline gets updated automatically when called
        reward=self.baseline(reward)

        sender_loss=torch.sum((-torch.log(a_sender[np.arange(len(message)),message]) * reward))
        receiver_loss=(-torch.log(a_receiver[answer]) * reward)


        self.losstime+=time.time()-lt
        bt=time.time()
        
        #reset optimizer and calculate gradients/weight upates
        self.sender_optimizer.zero_grad()
        sender_loss.backward() 

        self.receiver_optimizer.zero_grad()
        receiver_loss.backward()
        
        self.backtime+=time.time()-bt


        ut=time.time()
        #apply updates (no batching)
        self.sender_optimizer.step()
        self.receiver_optimizer.step()
        self.updatetime+=time.time()-ut
                
        self.n_steps+=1
        
        
class REINFORCEGS:
    """
    Extremely specific REINFORCE implementation, so far entails environment and everything and requires gs-agents
    """
    
    def __init__(self, dataset, agent, trainset, lr=1e-2, device="cpu", baseline=Baseline(), logging=[], n_steps=0, verbosity=-1, eval_steps=-1):
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
        message=self.agent.message.copy()#maybe we will need it
        
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
        
        # rewards are given per each correct one        
        
        reward= 1.0 if all(np.equal(one_hot(answer,self.n_values).flatten(),level)) else -1.0
        #sum([1.0 if answer[i]==self.dataset[level][i] else -1.0/self.n_attributes for i in range(len(answer))])#why are python list operators not vectorized to begin with?

        self.logging.append(reward)
        
        #reward is averaged, baseline gets updated automatically when called
        reward=self.baseline(reward)

        loss=sum([(-torch.log(action[i,answer[i]]) * reward) for i in range(len(answer))])


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