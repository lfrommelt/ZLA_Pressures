# Data
Generate artifical compositional data according to power law distributions. Train test split wrt. zero-shot compositionality.

So far two data distributions are implemented

## Local Zipf
The values for individual attributes are Zipf distributed. Afterwards these attributes are combined with each other as a cartesian product. This results in the distribution over the resulting Dataset to have plateaus when different codes have the same frequency rank (e.g. frequency-rank([1,2,3]) == frequency-rank([3,2,1]).
![absolute frequency as function of rank](examples/local_zipf.png)
![absolute frequency as function of actual codes, only partial for visibility reasons](examples/local_zipf_ordering.png)

## Global (unordered) Zipf
The overall dataset is Zipf distributed. Frequency ranks are not assigned wrt. specific symbols, though. Non-deterministic but n_dataset parameter because of sampling.
![absolute frequency as function of rank](examples/global_unsorted_zipf.png)
![absolute frequency as function of actual codes, only partial for visibility reasons](examples/global_unsorted_zipf_ordering.png)


## todo: Global (ordered) Zipf
Wrt. frequency it should be:
$[0,0,0] \gt [0,0,1] \gt [0,1,0] \gt ... [3,3,2] \gt [3,3,3]$
