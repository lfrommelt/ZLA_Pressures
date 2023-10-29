from evaluation.reference_distributions import *
from training.agents import GSPolicyNet
import matplotlib.pyplot as plt
import numpy as np
import json
import torch

def one_hot(values, n_values):
    '''todo: put in util'''
    one_hot_vector = np.eye(n_values)[values]
    return one_hot_vector.astype("int")


def evaluate(alphabet_size, dataset, agents, reference_distributions = [OptimalCoding], condition="fixed lenght", device="cpu"):
    #todo: assertions for same shape of messages
    for dist in reference_distributions:
        instance = dist(alphabet_size)
        plt.plot(np.arange(len(dataset)), instance.get_sequence(max_rank=len(dataset)+1), label=dist.__name__)
        
    for name in agents:
        with open("dump/"+str(name)+".json", "rt") as file:
            config=json.load(file)
            
        agent=GSPolicyNet(config["N_ATTRIBUTES"]*config["N_VALUES"], (config["MESSAGE_LENGTH"], config["ALPHABET_SIZE"]), (config["N_ATTRIBUTES"], config["N_VALUES"]))
        agent.load_state_dict(torch.load("dump/"+str(name)+".pt").state_dict())
        
        lengths=np.zeros(len(dataset))
        
        if "fix" in condition:
            messages = np.zeros((len(dataset), len(agent.message)))
            for i in range(len(messages)):
                state=torch.from_numpy(one_hot(dataset[i],max(dataset.flatten())+1).flatten())
                state=state.type(torch.float).to(device)
                #performing one forwar pass stores the respective message, should be changed so that forward directly returns message (required for aux loss anyways)
                agent(state)
                messages[i]=agent.message
                no_comm=np.argmax(np.bincount(messages[i].flatten().astype("int")))
                lengths[i]=len(messages[i][messages[i]!=no_comm])
        
        plt.plot(np.arange(len(lengths)), lengths, label=agent.__class__.__name__)
    
    plt.legend()
    plt.show()
                
            