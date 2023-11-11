from evaluation.reference_distributions import *
from training.agents import GSPolicyNet, SRGSPolicyNet
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import os
import pandas as pd
import re


def list_runs(folder_path="dump", start=0, stop=-1, avg_window=100):
    """
    Reads all JSON files in a specified folder and creates a DataFrame.

    Parameters:
    - folder_path (str): The path to the folder containing JSON files.

    Returns:
    - pd.DataFrame: A DataFrame containing data from all JSON files.
    """
    # List to store individual DataFrames from each JSON file
    data_frames = []

    filenames=[]
    
    for name in os.listdir(folder_path):
        if name.endswith(".json"):
            filenames.append(name)
    # Iterate through each file in the specified folder
    if stop>0:
        stop+=1
    else:
        stop=len(filenames)
    for filename in sorted(filenames[start:stop], key = lambda x: int(x[:-5])):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a JSON file
        if filename.endswith(".json"):
            # Read JSON file and create a DataFrame
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                #print(data.get("Note", json_file))
                df = pd.json_normalize(data)  # Flatten nested JSON structures if any
                df["Name"]=filename
                values=data["log"]
                df["best"]=max([np.mean(values[i:i+avg_window]) for i in range(0,len(values)-avg_window, avg_window)])
                data_frames.append(df)

    # Concatenate all DataFrames into a single DataFrame
    if data_frames:
        result_df = pd.concat(data_frames, ignore_index=True)
        return result_df
    else:
        print("No JSON files found in the specified folder.")
        return None
    
    
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
            
        the_class = re.match(r".*'.*agents.(.*)'", "<class 'training.agents.SRGSPolicyNet'>")[1]

        agent=eval(the_class)(config["N_ATTRIBUTES"]*config["N_VALUES"], (config["MESSAGE_LENGTH"], config["ALPHABET_SIZE"]), (config["N_ATTRIBUTES"], config["N_VALUES"]))
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
                if not i%200:
                    print(messages[i])
                
            no_comm=np.argmax(np.bincount(messages.flatten().astype("int")))
            print(f"agent {agent.__class__.__name__} no_comm: {no_comm}")
            print(np.bincount(messages.flatten().astype("int"), minlength=config["ALPHABET_SIZE"]))
            for i in range(len(messages)):
                lengths[i]=len(messages[i][messages[i]!=no_comm])
        
        plt.plot(np.arange(len(lengths)), lengths, label=agent.__class__.__name__)
    
    plt.legend()
    plt.show()   
    
def succes_per_target(alphabet_size, dataset, agents, avg_window=10, condition="fixed lenght", device="cpu", classification=True):
    #todo: assertions for same shape of messages
        
    for name in agents:
        with open("dump/"+str(name)+".json", "rt") as file:
            config=json.load(file)
            
        the_class = re.match(r".*'.*agents.(.*)'", "<class 'training.agents.SRGSPolicyNet'>")[1]

        agent=eval(the_class)(config["N_ATTRIBUTES"]*config["N_VALUES"], (config["MESSAGE_LENGTH"], config["ALPHABET_SIZE"]), (config["N_ATTRIBUTES"], config["N_VALUES"]), classification=classification)
        agent.load_state_dict(torch.load("dump/"+str(name)+".pt", map_location=torch.device(device)).state_dict())
        
        successes=np.zeros(len(dataset.dataset))
        answer_counts=np.zeros(len(dataset.dataset))
        
        np_dset=dataset.dataset.numpy()
        
        if "fix" in condition:
            
            if classification:#check for classification"SR" in config["Agent"]:
                answers = np.zeros(len(dataset.dataset))
            else:
                answers = np.zeros((len(dataset.dataset), dataset.n_attributes))
                
                
            for i in range(len(answers)):
                #state=torch.from_numpy(dataset.dataset[i].flatten())
                #state=state.type(torch.float).to(device)
                #performing one forwar pass stores the respective message, should be changed so that forward directly returns message (required for aux loss anyways)
                answer=agent(dataset.dataset[i])
                
                answers[i]=answer.cpu().detach().argmax(dim=-1)
                
                if classification:
                    successes[i]=answers[i]==i
                else:
                    #top 10 ugliest lines of code I have ever produced....
                    successes[i]=torch.all(torch.nn.functional.one_hot(torch.from_numpy(answers[i]).long(), num_classes=dataset.n_values)==dataset.dataset[i].view((dataset.n_attributes, dataset.n_values)))
                    
                answer_counts[np.argmax(np_dset==answer)]+=1
                if not i%200:
                    print(answers[i])
                
            
            print(f"agent {agent.__class__.__name__} max_out: {np.argmax(answer_counts)}")
        
        plt.plot(np.arange(len(answer_counts)), answer_counts, label=agent.__class__.__name__)
        plt.title("Answer counts")
        plt.legend()
        plt.show()
        
        print(f"{sum(successes)} successful items")
        plt.plot(np.arange(len(successes)/avg_window), [np.mean(successes[i:i+avg_window]) for i in range(0,len(successes)-avg_window+1, avg_window)], label=agent.__class__.__name__)
        plt.legend()
        plt.show()
                
            