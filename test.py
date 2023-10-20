from evaluation.reference_distributions import OptimalCoding
from data.data import Dataset

if __name__ == "__main__" or True:
    #plot from the paper
    dset = Dataset(3, 4, distribution = "local_values", data_size_scale=10)
    dset.plot()
    dset.plot(explicit=True)
    dset = Dataset(3, 4, distribution = "unordered", data_size_scale=1000, distribution_param=1.5)
    dset.plot()
    dset.plot(explicit=True)