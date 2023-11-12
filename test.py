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
from training.reinforce import Baseline, REINFORCEGS, Supervised, SRSupervised, SRREINFORCE
from training.loss import LengthLoss, LeastEffortLoss, DirichletLoss

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


LEARNING_RATE=1e-3#1e-4
ALPHABET_SIZE=40
MESSAGE_LENGTH=30
N_EPOCHS=2000
N_TRAINSET=1000#not implemented
N_TESTSET=100#not implemented
EVAL_STEP=1000#00#00
N_DISTRACTORS=-1
N_ATTRIBUTES=3
N_VALUES=10
VERBOSE=EVAL_STEP
DEVICE="cpu"#"cuda"
N_STEPS=1000000

AGENT=GSPolicyNet
ALGORITHM=Supervised
CLASSIFICATION=False#False#True#False#True

AUX_LOSSES=[LeastEffortLoss()]#[DirichletLoss(alphabet_size=ALPHABET_SIZE, differentiable=True)]#LeastEffortLoss()]#[LengthLoss(0.0)]
BETA1=45
BETA2=10

CONTINUE=0#"28"

trainrun=0

if not CONTINUE:
    dataloader = Dataloader(N_ATTRIBUTES, N_VALUES, device=DEVICE)
    #indexing array, could later be used for train-test-split
    trainset = np.arange(len(dataloader))

    NOTE=input("describe the run: ")#"lr tuning for mixed length (and classifiction)"
    agent = AGENT(N_ATTRIBUTES*N_VALUES, (MESSAGE_LENGTH, ALPHABET_SIZE), (N_ATTRIBUTES,N_VALUES), device=DEVICE, classification=CLASSIFICATION).to(DEVICE)
    start=0
    reinforce=ALGORITHM(dataloader, agent, trainset, lr=LEARNING_RATE, device=DEVICE, aux_losses=AUX_LOSSES, classification=CLASSIFICATION)


else:
    with open('dump/'+CONTINUE+'.json') as json_file:
        data = json.load(json_file)
    locals().update(data)
    dataloader = Dataloader(N_ATTRIBUTES, N_VALUES, device=DEVICE)
    #indexing array, could later be used for train-test-split
    trainset = np.arange(len(dataloader))


    start=len(log)
    agent = torch.load("dump/"+CONTINUE+".pt")
    reinforce=ALGORITHM(dataloader, agent, trainset, lr=LEARNING_RATE, device=DEVICE, aux_losses=AUX_LOSSES, classification=CLASSIFICATION, lr_listener=LR_LISTEN)
    reinforce.logging=log
    trainrun=int(CONTINUE)



alpha_func = lambda x: x**BETA1/BETA2
#should become a function `training()` that is parameterized with the hyperparameters
'''
sender_policy=SRPolicyNet(N_ATTRIBUTES*N_VALUES,(MESSAGE_LENGTH,ALPHABET_SIZE),device=DEVICE).to(DEVICE)
receiver_policy=SRPolicyNet(ALPHABET_SIZE*MESSAGE_LENGTH+N_ATTRIBUTES*N_VALUES*(N_DISTRACTORS+1),(1,N_DISTRACTORS+1),device=DEVICE).to(DEVICE)
'''
#agent = GSPolicyNetLSTM(N_ATTRIBUTES*N_VALUES, (MESSAGE_LENGTH, ALPHABET_SIZE), (N_ATTRIBUTES,N_VALUES), device=DEVICE).to(DEVICE)

average_logging=[]


#bit weird, but we iterate over the whole dataset instead of sampling from it. Should that be changed?

#generator for sampling
sampler=dataloader.sampler(index=True)

#save agents and log under new name, name is simply the number of the trainrun
# overnight: BETA1, BETA2 in ((45,10),(45,15),(45,20),(45,30))

for i in tqdm(range(start, N_STEPS)):
    level=next(sampler)

    reinforce.step(level)

    #again future training function
    if not (i+1)%EVAL_STEP:
        
        if not trainrun:
                names = os.listdir("dump")
                names=filter(lambda x: x.endswith("pt"),os.listdir("dump"))
                names = [re.match(r"(\d*).pt", name)[1] for name in names]
                trainrun=int(max(names, key = lambda x: int(x)))+1
                
        average_logging.append(sum(reinforce.logging[-EVAL_STEP:])/EVAL_STEP)

        alpha = alpha_func(average_logging[-1])
        print("Step",i)
        print("Average, raw reward", average_logging[-1])

        if AUX_LOSSES:
            print("alpha:", AUX_LOSSES[0].alpha)

            AUX_LOSSES[0].alpha=alpha

        '''
        print("agenttime:", reinforce.agenttime,"\n",
              "losstime:    ", reinforce.losstime,"\n",
              "backtime:    ", reinforce.backtime,"\n",
              "upatetime:   ", reinforce.updatetime),"\n",'''

        reinforce.agenttime=0
        reinforce.losstime=0
        reinforce.backtime=0
        reinforce.updatetime=0

        print(f"saving as: {trainrun}.pt")

        if len(average_logging)>1:
            if average_logging[-1]>average_logging[-2]:
                torch.save(agent,os.path.normpath("dump/"+str(trainrun)+".pt"))#todo: state_ict

        config_and_log = {"Losses":[str(loss) for loss in AUX_LOSSES],
                          "Agent": str(agent),
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
                          "Note": NOTE,
                          "beta1": BETA1,
                          "beta2": BETA2,
                          "classifier": CLASSIFICATION}


        with open(os.path.normpath("dump/"+str(trainrun)+".json"), "w") as file:
            json.dump(config_and_log, file)
    #file.write(LEARNING_RATE, ALPHABET_SIZE, N_EPOCHS, N_TRAINSET, N_TESTSET, N_DISTRACTORS, N_ATTRIBUTES, MESSAGE_LENGTH, N_VALUES, DEVICE,

plt.plot(np.arange(len(average_logging)), average_logging)
plt.show()
