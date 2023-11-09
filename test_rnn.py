import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re
import json

from evaluation.reference_distributions import OptimalCoding
from data.data import Dataloader
from training.agents import SRPolicyNet, GSPolicyNet, GSPolicyNetLSTM#will happen in training script
from training.reinforce import Baseline, REINFORCEGS, Supervised
from training.loss import LengthLoss

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

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

LEARNING_RATE=1e-3#1e-2#good results so far
ALPHABET_SIZE=40
MESSAGE_LENGTH=30
N_EPOCHS=2000
N_TRAINSET=1000#not implemented
N_TESTSET=100#not implemented
EVAL_STEP=5000#00#00
N_DISTRACTORS=-1
N_ATTRIBUTES=3
N_VALUES=10
VERBOSE=EVAL_STEP
DEVICE="cpu"#"cuda"
N_STEPS=60000

AGENT=GSPolicyNetLSTM
ALGORITHM=Supervised
CLASSIFICATION=True

AUX_LOSSES=[LengthLoss(1.0/MESSAGE_LENGTH)]
NOTE=input("describe the run")#"lr tuning for mixed length (and classifiction)"

dataloader = Dataloader(N_ATTRIBUTES, N_VALUES, device=DEVICE)

#should become a function `training()` that is parameterized with the hyperparameters
'''
sender_policy=SRPolicyNet(N_ATTRIBUTES*N_VALUES,(MESSAGE_LENGTH,ALPHABET_SIZE),device=DEVICE).to(DEVICE)
receiver_policy=SRPolicyNet(ALPHABET_SIZE*MESSAGE_LENGTH+N_ATTRIBUTES*N_VALUES*(N_DISTRACTORS+1),(1,N_DISTRACTORS+1),device=DEVICE).to(DEVICE)
'''
#overnight: N_STEPS=600000, LEARNING_RATE in [5e-4, 5e-5]
for LEARNING_RATE in [1e-3, 1e-4]:
    #agent = GSPolicyNetLSTM(N_ATTRIBUTES*N_VALUES, (MESSAGE_LENGTH, ALPHABET_SIZE), (N_ATTRIBUTES,N_VALUES), device=DEVICE).to(DEVICE)
    agent = AGENT(N_ATTRIBUTES*N_VALUES, (MESSAGE_LENGTH, ALPHABET_SIZE), (N_ATTRIBUTES,N_VALUES), device=DEVICE).to(DEVICE)

    average_logging=[]

    #indexing array, could later be used for train-test-split
    trainset = np.arange(len(dataloader))

    reinforce=ALGORITHM(dataloader, agent, trainset, lr=LEARNING_RATE, device=DEVICE, aux_losses=AUX_LOSSES, classification=CLASSIFICATION)
    #bit weird, but we iterate over the whole dataset instead of sampling from it. Should that be changed?
    
    #generator for sampling
    sampler=dataloader.sampler(index=True)

    #save agents and log under new name, name is simply the number of the trainrun
    names = os.listdir("dump")
    names=filter(lambda x: x.endswith("pt"),os.listdir("dump"))
    names = [re.match(r"(\d*).pt", name)[1] for name in names]
    trainrun=int(max(names, key = lambda x: int(x)))+1
            
        
    for i in tqdm(range(N_STEPS)):
        level=next(sampler)

        reinforce.step(level)

        #again future training function
        if not (i+1)%EVAL_STEP:
            average_logging.append(sum(reinforce.logging[-EVAL_STEP:])/EVAL_STEP)

            print("Step",i)
            print("Average, raw reward", average_logging[-1])


            print("agenttime:", reinforce.agenttime,"\n",
                  "losstime:    ", reinforce.losstime,"\n",
                  "backtime:    ", reinforce.backtime,"\n",
                  "upatetime:   ", reinforce.updatetime),"\n",

            reinforce.agenttime=0
            reinforce.losstime=0
            reinforce.backtime=0
            reinforce.updatetime=0
            
            print(f"saving as: {trainrun}.pt")

            if len(average_logging)>1:
                if average_logging[-1]>average_logging[-2]:
                    torch.save(agent,os.path.normpath("dump/"+str(trainrun)+".pt"))#todo: state_ict

            config_and_log = {"Losses":[str(loss) for loss in AUX_LOSSES],
                              "Algorithm": ALGORITHM.__name__,
                              "LEARNING_RATE": LEARNING_RATE,
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
                              "log": reinforce.logging,
                              "Note": NOTE,}


            with open(os.path.normpath("dump/"+str(trainrun)+".json"), "w") as file:
                json.dump(config_and_log, file)
        #file.write(LEARNING_RATE, ALPHABET_SIZE, N_EPOCHS, N_TRAINSET, N_TESTSET, N_DISTRACTORS, N_ATTRIBUTES, MESSAGE_LENGTH, N_VALUES, DEVICE,

plt.plot(np.arange(len(average_logging)), average_logging)
plt.show()
