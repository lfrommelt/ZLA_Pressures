from evaluation.reference_distributions import *
from training.agents import GSPolicyNet
import matplotlib.pyplot as plt
import numpy as np
import json
import torch

def one_hot(values, n_values):
    '''todo: put in util'''
    values=values.cpu().numpy()
    print(values)
    one_hot_vector = np.eye(n_values)[values]
    return one_hot_vector.astype("int")

def training_curve(agents, avg_window=100, subset=(0,-1), label=0):
    """
    Plots rewards during training for a trainrun saved in dump/
    Note: not running mean (because padding), but averaged bins
    caveat: last bin could be really small...

    Parameters
    ----------
        agents : list(int)
            name of the json in dump
        avg_window : int
            rewards will be averaged over this amount of steps
    """
    for agent in agents:
        with open("dump/"+str(agent)+".json", "rt") as file:
            config=json.load(file)
        values = config["log"][subset[0]:subset[1]]
        averages=[np.mean(values[i:i+avg_window]) for i in range(0,len(values)-avg_window, avg_window)]
        if label:
            name = config[label]
        else:
            name = str(agent)
        plt.plot(range(0,len(values)-avg_window,avg_window),averages,label=name)
    plt.legend()
    plt.show()
            
            
def evaluate(alphabet_size, dataset, agents, reference_distributions = [OptimalCoding], condition="fixed lenght", device="cpu"):
    #todo: assertions for same shape of messages
    for dist in reference_distributions:
        instance = dist(alphabet_size)
        plt.plot(np.arange(len(dataset.dataset)), instance.get_sequence(max_rank=len(dataset.dataset)+1), label=dist.__name__)
        
    for name in agents:
        with open("dump/"+str(name)+".json", "rt") as file:
            config=json.load(file)
            
        agent=GSPolicyNet(config["N_ATTRIBUTES"]*config["N_VALUES"], (config["MESSAGE_LENGTH"], config["ALPHABET_SIZE"]), (config["N_ATTRIBUTES"], config["N_VALUES"]))
        agent.load_state_dict(torch.load("dump/"+str(name)+".pt", map_location=torch.device(device)).state_dict())
        
        lengths=np.zeros(len(dataset.dataset))
        
        if "fix" in condition:
            messages = np.zeros((len(dataset.dataset), len(agent.message)))
            for i in range(len(messages)):
                #state=torch.from_numpy(dataset.dataset[i].flatten())
                #state=state.type(torch.float).to(device)
                #performing one forwar pass stores the respective message, should be changed so that forward directly returns message (required for aux loss anyways)
                agent(dataset.dataset[i])
                messages[i]=agent.message.cpu().detach().argmax(dim=-1)
                
            no_comm=np.argmax(np.bincount(messages.flatten().astype("int")))
            print(f"agent {agent.__class__.__name__} no_comm: {no_comm}")
            print(np.bincount(messages.flatten().astype("int")))
            for i in range(len(messages)):
                lengths[i]=len(messages[i][messages[i]!=no_comm])
        
        plt.plot(np.arange(len(lengths)), lengths, label=agent.__class__.__name__)
    
    plt.legend()
    plt.show()
                
            