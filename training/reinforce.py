import torch
import tqdm
from sklearn.datasets import load_digits
import numpy as np

ALPHABET_SIZE=11
N_EPOCHS=4000
N_TRAINSET=1000
N_TESTSET=100
EVAL_STEP=N_EPOCHS/10#10000

class PolicyNet(torch.nn.Module):

    def __init__(self, in_shape, out_shape):
        super().__init__()

        self.linear1 = torch.nn.Linear(in_shape, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, out_shape)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    

class Baseline:
    
    def __init__(self):
        self.rewards=[0,]
        
    def __call__(self, reward):
        self.rewards.append(reward)
        return (reward-np.mean(self.rewards))/np.std(self.rewards)
        
def entropy(probs):
    '''wtf pytorch...'''
    assert all(torch.isclose(torch.sum(probs),torch.tensor([1.0]))), probs
    assert not 0.0 in probs, probs
    return -torch.sum(probs*torch.log(probs))

device="cpu"

sender_policy=PolicyNet(64,ALPHABET_SIZE).to(device)
receiver_policy=PolicyNet(ALPHABET_SIZE+3*64,3).to(device)

dataset=load_digits()
trainset=np.arange(0,N_TRAINSET)
testset=np.arange(N_TRAINSET,N_TRAINSET+N_TESTSET)

sender_optimizer = torch.optim.Adam(sender_policy.parameters(), lr=1e-5)
receiver_optimizer = torch.optim.Adam(receiver_policy.parameters(), lr=1e-5)

logging=[]
overall_logging=[]
baseline=Baseline()


for i in tqdm.tqdm(range(N_EPOCHS)):
    np.random.shuffle(trainset)
    for level in trainset:
        distractors=np.zeros((3,64))
        distractors[0]=dataset["data"][level]
        i=0
        while i<2:
            candidate=np.random.choice(trainset)
            if not candidate in distractors:
                i+=1
                distractors[i]=dataset["data"][candidate]
                
        #level=np.random.choice(trainset)
        s_sender=torch.from_numpy(dataset["data"][level]).to(device)
        s_sender=s_sender.type(torch.float)
        a_sender=sender_policy(s_sender)

        message=np.random.choice(np.arange(ALPHABET_SIZE), p=a_sender.detach().cpu().numpy())
        

        s_receiver=np.array([1.0 if i==message else 0 for i in range(ALPHABET_SIZE)])#torch.nn.functional.one_hot(torch.Tensor(message), num_classes=ALPHABET_SIZE)
        
        shuffle=np.arange(3,dtype="int")
        np.random.shuffle(shuffle)
        distractors=np.array(distractors)
        
        s_receiver=torch.from_numpy(np.concatenate([*distractors[shuffle],s_receiver])).type(torch.float).to(device)
        a_receiver=receiver_policy(s_receiver)

        answer=np.random.choice(np.arange(3), p=a_receiver.detach().cpu().numpy())

        reward= 1.0 if shuffle[answer]==0 else -1.0

        logging.append(reward)

        reward=baseline(reward)
        
        sender_loss=(-torch.log(a_sender[message]) * reward)
        receiver_loss=(-torch.log(a_receiver[answer]) * reward)
        sender_loss.backward()
        receiver_loss.backward()

    sender_optimizer.zero_grad()
    sender_optimizer.step() 
    
    receiver_optimizer.zero_grad()
    receiver_optimizer.step()
    overall_logging.append(sum(logging)/len(trainset))
    logging=[]
    
    if not (i+1)%EVAL_STEP:
        print(i)
        print(sum(logging)/EVAL_STEP)
        logging=[]
        
plt.plot(np.arange(len(overall_logging)), overall_logging)
plt.show()