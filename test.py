import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re
import json

from evaluation.reference_distributions import OptimalCoding
from data.data import Dataset
from training.agents import SRPolicyNet, GSPolicyNet#will happen in training script
from training.reinforce import REINFORCE, Baseline, REINFORCEGS

#old stuff
'''
def test_data():
    #plot/params from the paper with our dataset
    dset = Dataset(3, 4, distribution = "local_values", data_size_scale=10)
    dset.plot()
    dset.plot(explicit=True)
    dset = Dataset(3, 4, distribution = "unordered", data_size_scale=1000, distribution_param=1.5)
    dset.plot()
    dset.plot(explicit=True)'''

LEARNING_RATE=1e-4#1e-2#good results so far
ALPHABET_SIZE=11
N_EPOCHS=5000
N_TRAINSET=1000#not implemented
N_TESTSET=100#not implemented
EVAL_STEP=500#00
N_DISTRACTORS=-1
N_ATTRIBUTES=3
MESSAGE_LENGTH=5
N_VALUES=4
VERBOSE=EVAL_STEP
DEVICE="cpu"#"cuda"

for LEARNING_RATE in [1e-4, 1e-5]:
    dset = Dataset(N_ATTRIBUTES, N_VALUES, distribution = "local_values", data_size_scale=10)

    #should become a function `training()` that is parameterized with the hyperparameters
    '''
    sender_policy=SRPolicyNet(N_ATTRIBUTES*N_VALUES,(MESSAGE_LENGTH,ALPHABET_SIZE),device=DEVICE).to(DEVICE)
    receiver_policy=SRPolicyNet(ALPHABET_SIZE*MESSAGE_LENGTH+N_ATTRIBUTES*N_VALUES*(N_DISTRACTORS+1),(1,N_DISTRACTORS+1),device=DEVICE).to(DEVICE)
    '''

    agent = GSPolicyNet(N_ATTRIBUTES*N_VALUES, (MESSAGE_LENGTH, ALPHABET_SIZE), (N_ATTRIBUTES,N_VALUES)).to(DEVICE)

    average_logging=[]

    dataset = dset.dataset.copy()
    #indexing array, could later be used for train-test-split
    trainset = np.arange(len(dataset))

    reinforce=REINFORCEGS(dataset, agent, trainset, lr=LEARNING_RATE)
    #bit weird, but we iterate over the whole dataset instead of sampling from it. Should that be changed?
    for epoch in tqdm(range(N_EPOCHS)):
        np.random.shuffle(trainset)

        #new part
        for level in trainset:
            reinforce.step(level)

        #again future training function
        average_logging.append(sum(reinforce.logging[-len(trainset):])/len(trainset))

        if not (epoch+1)%EVAL_STEP:
            print("Epoch",epoch)
            print("Average, raw reward", average_logging[-1])


            print("agenttime:", reinforce.agenttime,"\n",
                  "losstime:    ", reinforce.losstime,"\n",
                  "backtime:    ", reinforce.backtime,"\n",
                  "upatetime:   ", reinforce.updatetime),"\n",

            reinforce.agenttime=0
            reinforce.losstime=0
            reinforce.backtime=0
            reinforce.updatetime=0
    
    #save agents and log under new name, name is simply the number of the trainrun
    names = sorted(os.listdir("dump"), reverse=True)
    
    #find out how many trainruns are alreay saved on harddrive
    trainrun = 0
    for name in names:
        if name.endswith(".pt"):
            trainrun=re.match(r"(\d*).pt", name)[1]
            trainrun=str(int(trainrun)+1)
            break

    torch.save(agent,os.path.normpath("dump/"+str(trainrun)+".pt"))
    
    config_and_log = {"LEARNING_RATE": LEARNING_RATE,
                      "ALPHABET_SIZE": ALPHABET_SIZE,
                      "N_EPOCHS": N_EPOCHS,
                      "N_TRAINSET": N_TRAINSET,
                      "N_TESTSET": N_TESTSET,
                      "EVAL_STEP": EVAL_STEP,
                      "N_DISTRACTORS": N_DISTRACTORS,
                      "N_ATTRIBUTES": N_ATTRIBUTES,
                      "MESSAGE_LENGTH": MESSAGE_LENGTH,
                      "N_VALUES": N_VALUES,
                      "DEVICE": DEVICE,
                      "agenttime:": reinforce.agenttime,
                      "losstime:    ": reinforce.losstime,
                      "backtime:    ": reinforce.backtime,
                      "upatetime:   ": reinforce.updatetime,
                      "log": reinforce.logging}
    
    
    with open(os.path.normpath("dump/"+str(trainrun)+".json"), "w") as file:
        json.dump(config_and_log, file)
        #file.write(LEARNING_RATE, ALPHABET_SIZE, N_EPOCHS, N_TRAINSET, N_TESTSET, N_DISTRACTORS, N_ATTRIBUTES, MESSAGE_LENGTH, N_VALUES, DEVICE,
    
    plt.plot(np.arange(len(average_logging)), average_logging)
    plt.show()