# ZLA_Pressures
Implementation of project work for AMMAC

# Next Steps
- fixed length setting: check if sparse rewards (reward iff ALL output values match) is better
- refine fixed length setting
	- test for alphabet size from paper
	- we need 100% (avg reward=3.0), maybe first try sparse rewards
	- "no communication" can be used for conveying information by positional encoding, shuffling message should get rid of that problem, but probably requires completely new hyperparams
- implement mixed length setting
	- RNNs for generating message as well as "reading" it
	- check if gumbel softmax is possible
	- include mixed length in evluation
- implement aux losses and check how they perform regarding task success:
	- length penalty (penalize a priori defined "no communication" token)
	- length penalty curriculum (define sucess rate (only perfect reconstructions vs. partially correct reconstructions), find nice function alpha/try and induce it from paper fig)
	- CE(y_pred; gumbel-softmax(y_pred).detach())), with y_pred being the probabilities
		- is there a hyperparam for scaling?
		- does it even work with gs?
		- include in reward baseline? (i.e. `baseline(reward+loss)`)
	- CE(y_pred; dirichlet(old_params, argmax(message)))
		- same conciderations as above
		- look in paper to make sure that this is actually how they did it
- misc:
	- check out embedding layers (tradeoff: runtime vs. accuracy??) between vector and networks as well as between message and networks
	- Monkey typing reference distribution
	- eval plot for comparing training curves (in .json @ "log")
	- refactor main content of test.py to training/training.py
	- refactor util functions that are here and there, so they are in some common utils.py
	- add checkpoints to training in case it crashes
