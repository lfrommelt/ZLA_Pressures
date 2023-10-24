import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from evaluation.reference_distributions import OptimalCoding
from data.data import Dataset
from training.agents import SRPolicyNet#will happen in training script
from training.reinforce import REINFORCE, Baseline

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

LEARNING_RATE=1e-2#good results so far
ALPHABET_SIZE=11
N_EPOCHS=15#00
N_TRAINSET=1000
N_TESTSET=100
EVAL_STEP=5#todo log all data and remove this
N_DISTRACTORS=1
N_ATTRIBUTES=3
MESSAGE_LENGTH=5
N_VALUES=4
VERBOSE=EVAL_STEP
DEVICE="cpu"#"cuda"

dset = Dataset(N_ATTRIBUTES, N_VALUES, distribution = "local_values", data_size_scale=10)

#should become a function `training()` that is parameterized with the hyperparameters
sender_policy=SRPolicyNet(N_ATTRIBUTES*N_VALUES,(MESSAGE_LENGTH,ALPHABET_SIZE),device=DEVICE).to(DEVICE)
receiver_policy=SRPolicyNet(ALPHABET_SIZE*MESSAGE_LENGTH+N_ATTRIBUTES*N_VALUES*(N_DISTRACTORS+1),(1,N_DISTRACTORS+1),device=DEVICE).to(DEVICE)



sender_optimizer = torch.optim.Adam(sender_policy.parameters(), lr=LEARNING_RATE)
receiver_optimizer = torch.optim.Adam(receiver_policy.parameters(), lr=LEARNING_RATE)

average_logging=[]

dataset = dset.dataset.copy()
#indexing array, could later be used for train-test-split
trainset = np.arange(len(dataset))

reinforce=REINFORCE(dataset, sender_policy, receiver_policy, trainset, lr=LEARNING_RATE)
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
        
        
        print("sendertime:  ", reinforce.sendertime,"\n",
              "listenertime:", reinforce.listenertime,"\n",
              "losstime:    ", reinforce.losstime,"\n",
              "backtime:    ", reinforce.backtime,"\n",
              "upatetime:   ", reinforce.updatetime),"\n",

        reinforce.sendertime=0
        reinforce.listenertime=0
        reinforce.losstime=0
        reinforce.backtime=0
        reinforce.updatetime=0


plt.plot(np.arange(len(average_logging)), average_logging)
plt.show()