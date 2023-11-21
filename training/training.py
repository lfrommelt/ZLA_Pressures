from tqdm import tqdm
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.data import Dataloader
from training.agents import SRPolicyNet, GSPolicyNet, GSPolicyNetLSTM
from training.algorithms import Baseline, REINFORCEGS, Supervised, SRSupervised, SRREINFORCE
from training.loss import LengthLoss, LeastEffortLoss, DirichletLoss

'''
default parameters as constants
'''
# training
LEARNING_RATE=1e-3#1e-4
N_STEPS=1000000
ALGORITHM=Supervised

# message properties
ALPHABET_SIZE=40
MESSAGE_LENGTH=30

# task
N_ATTRIBUTES=3
N_VALUES=10

# condition
AGENT=GSPolicyNet
AUX_LOSSES=[LeastEffortLoss()]
BETA1=45
BETA2=10

# test for generalization properties
N_TRAINSET=1000#not implemented
N_TESTSET=100#not implemented

# misc
DEVICE="cpu"#"cuda"
EVAL_STEP=1000#print current success rate and do checkpoint
VERBOSE=EVAL_STEP
CONTINUE=0#enables to continute training from a checkpoint

def training(
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    algorithm=ALGORITHM,
    alphabet_size=ALPHABET_SIZE,
    message_length=MESSAGE_LENGTH,
    n_attributes=N_ATTRIBUTES,
    n_values=N_VALUES,
    agent=AGENT,
    aux_losses=AUX_LOSSES,
    beta1=BETA1,
    beta2=BETA2,
    n_trainset=N_TRAINSET,
    n_testset=N_TESTSET,
    verbose=VERBOSE,
    device=DEVICE,
    eval_step=EVAL_STEP,
    continue_=CONTINUE,
    lr_listener=0,
    plot=False,
    classification=False
    ):
    """
    Description:
    ------------
    Train agents on a language game for a given amount of steps.

    Parameters:
    -----------
    learning_rate : float
        The learning rate for the training process.

    n_steps : int
        The number of training steps

    algorithm : training.algorithms.Algorithm
        The algorithm used for training. See training.algorithms for more information.

    alphabet_size : int
        The maximal size of the alphabet from wich messages will be generated.

    message_length : int
        The maximum length of the messages (mixed length condition) or the length of
        the messages (fixed length condition).

    n_attributes : int
        The number of attributes in the task.

    n_values : int
        The number of values in the task.

    agent : training.agents.Agent
        Class of the policy network used for the training.

    aux_losses : list([training.loss.Loss, ...])
        List of auxiliary losses.

    beta1 : int, optional
        Beta1 parameter for the alpha function.

    beta2 : int, optional
        Beta2 parameter for the alpha function.

    n_trainset : int, optional
        The size of the training set for generalization testing. Not implemented, yet.

    n_testset : int, optional
        The size of the test set for generalization testing. Not implemented, yet.

    verbose : str
        Verbosity level for logging. Legacy, but could become useful again. For now,
        you have to live with prints during training.

    device : str
        The torch device used for training. Recommendation: use cpu, cuda does not
        make things much faster with our small networks and no batching.

    eval_step : int
        Evaluation step for printing success rate and checkpointing

    continue_ : int
        Continue training from given checkpoint or fresh start if continue_ = 0

    plot : bool
        Flag to automatically plot train curve after training
        
    lr_listen : float
        Optionally set individual learning rate for listener for agents without
        Gumbel-Softmax.
        
    Returns:
    --------
    training_curve : mpl.plot
        Matplotlib plot of the training curve.
    """
    if not lr_listener:
        lr_listener=learning_rate
    trainrun=0
    
    if not continue_:
        dataloader = Dataloader(n_attributes, n_values, device=device)
        note = input("describe the run: ")
        agent = agent(n_attributes * n_values, (message_length, alphabet_size), (n_attributes, n_values), device=device, classification=classification).to(device)
        start = 0
        reinforce = algorithm(dataloader, agent, lr=learning_rate, device=device, aux_losses=aux_losses, classification=classification)

    else:
        with open('dump/' + continue_ + '.json') as json_file:
            data = json.load(json_file)
        locals().update(data)
        dataloader = Dataloader(n_attributes, n_values, device=device)
        # indexing array, could later be used for train-test-split
        #trainset = np.arange(len(dataloader))

        start = len(log)
        agent = torch.load("dump/" + continue_ + ".pt")
        reinforce = algorithm(dataloader, agent, lr=learning_rate, device=device, aux_losses=aux_losses, classification=classification, lr_listener=lr_listen)
        reinforce.logging = log
        trainrun = int(continue_)

    # will only be used if length_loss is in losses
    alpha_func = lambda x: x**beta1 / beta2

    average_logging = []

    # generator for sampling
    sampler = dataloader.sampler(index=True)

    for i in tqdm(range(start, n_steps)):
        level = next(sampler)

        reinforce.step(level)

        # again future training function
        if not (i + 1) % eval_step:
            # save agents and log under new name, name is simply the number of the trainrun
            if not trainrun:
                    names = os.listdir("dump")
                    names = filter(lambda x: x.endswith("pt"), os.listdir("dump"))
                    names = [re.match(r"(\d*).pt", name)[1] for name in names]
                    trainrun = int(max(names, key=lambda x: int(x))) + 1

            average_logging.append(sum(reinforce.logging[-eval_step:]) / eval_step)

            alpha = alpha_func(average_logging[-1])
            print("Step", i)
            print("Average, raw reward", average_logging[-1])

            if aux_losses:
                print("alpha:", aux_losses[0].alpha)

                aux_losses[0].alpha = alpha

            reinforce.agenttime = 0
            reinforce.losstime = 0
            reinforce.backtime = 0
            reinforce.updatetime = 0

            print(f"saving as: {trainrun}.pt")

            if len(average_logging) > 1:
                if average_logging[-1] > average_logging[-2]:
                    torch.save(agent, os.path.normpath("dump/" + str(trainrun) + ".pt"))  # todo: state_ict

            config_and_log = {"Losses": [str(loss) for loss in aux_losses],
                              "Agent": str(agent),
                              "Algorithm": algorithm.__name__,
                              "learning_rate": learning_rate,
                              "alphabet_size": alphabet_size,
                              "n_trainset": n_trainset,
                              "n_testset": n_testset,
                              "eval_step": eval_step,
                              "n_attributes": n_attributes,
                              "message_length": message_length,
                              "n_values": n_values,
                              "device": device,
                              "agenttime:": reinforce.agenttime,
                              "losstime:    ": reinforce.losstime,
                              "backtime:    ": reinforce.backtime,
                              "upatetime:   ": reinforce.updatetime,
                              "log": reinforce.logging,
                              "Note": note,
                              "beta1": beta1,
                              "beta2": beta2,
                              "classifier": classification}

            with open(os.path.normpath("dump/" + str(trainrun) + ".json"), "w") as file:
                json.dump(config_and_log, file)
        # file.write(learning_rate, alphabet_size, n_epochs, n_trainset, n_testset, n_distractors, n_attributes, message_length, n_values, device,

    training_curve = plt.plot(np.arange(len(average_logging)), average_logging)
    if plot:
        plt.show()
    return training_curve


"""
        NOTE=input("describe the run: ")
        agent = agent(n_attributes*n_values, (message_length, alphabet_size), (n_attributes,n_values), device=device, classification=classification).to(device)
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
        trainrun=int(continue_)

    #will only be used if LengthLoss is in losses
    alpha_func = lambda x: x**beta1/beta2

    average_logging=[]

    #generator for sampling
    sampler=dataloader.sampler(index=True)

    for i in tqdm(range(start, n_steps)):
        level=next(sampler)

        reinforce.step(level)

        #again future training function
        if not (i+1)%EVAL_STEP:
            #save agents and log under new name, name is simply the number of the trainrun
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

    training_curve = plt.plot(np.arange(len(average_logging)), average_logging)
    if plot:
        plt.show()
    return training_curve"""