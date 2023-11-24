# Training
Main folder, where most of the interesting stuff is happening. 

`agents` contains several architectures, that:
+ either have an individual architecture for sender and receiver or both are trained jointly via gumbel-softmax
+ either uses fully connected layers for fixed-length settings or LSTMs for mixe-length setting (LSTM conditions were not converging in feasible time, see Fig. 3 in submission pdf)

`algorithms` contains several training algorithms are inspired mostly by  [Chaabouni et al. (2019)](https://proceedings.neurips.cc/paper/2019/hash/31ca0ca71184bbdb3de7b20a51e88e90-Abstract.html) an that:
+ either use CE(label, output) as loss for gradient descent or CE(label, output) as (negative) reward.
+ eihter uses individual updates for Speaker an Listener, respectively, or trains them jointly
+ either uses the input as label, or the index of the input in the dataset ("classification", used like this in the paper but leas to terrible results)

`loss` contains the auxiliary losses that are used as ZLA-pressure as discussed in the submission pdf.

`training` contains a script for conducting experiments on the overall code based on given hyperparameters.