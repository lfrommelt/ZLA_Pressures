from evaluation.evaluation import evaluate
from data.data import Dataset

LEARNING_RATE=1e-4#1e-2#good results so far
ALPHABET_SIZE=11
N_EPOCHS=5000
N_TRAINSET=1000
N_TESTSET=100
EVAL_STEP=500#00
N_DISTRACTORS=-1
N_ATTRIBUTES=3
MESSAGE_LENGTH=5
N_VALUES=4
VERBOSE=EVAL_STEP
DEVICE="cpu"#"cuda"

dataset = Dataset(N_ATTRIBUTES, N_VALUES, distribution = "local_values", data_size_scale=10).dataset

evaluate(ALPHABET_SIZE, dataset, [1])