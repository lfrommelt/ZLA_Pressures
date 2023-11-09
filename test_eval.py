from evaluation.evaluation import evaluate
from data.data import Dataset, Dataloader

LEARNING_RATE=1e-4#1e-2#good results so far
ALPHABET_SIZE=40
N_EPOCHS=2000
N_TRAINSET=1000#not implemented
N_TESTSET=100#not implemented
EVAL_STEP=5000#00#00
N_DISTRACTORS=-1
N_ATTRIBUTES=3
MESSAGE_LENGTH=30
N_VALUES=10
VERBOSE=EVAL_STEP
DEVICE="cpu"#"cuda"
N_STEPS=200000


dataloader = Dataloader(N_ATTRIBUTES, N_VALUES, device=DEVICE)

evaluate(ALPHABET_SIZE, dataloader, [36, 37], device=DEVICE)
