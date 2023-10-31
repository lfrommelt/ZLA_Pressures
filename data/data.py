from data.utils import recursive_cartesian, zipf_distribution, one_hot
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

class Dataloader:
    '''
    For simplicity this is dataloader and dataset in one class together.
    '''
            
    def __init__(self, n_attributes, n_values, device="cpu"):
        
        #storing for convenience and explicitness
        self.n_values=n_values
        self.n_attributes=n_attributes
        
        data = recursive_cartesian(*[np.arange(n_values) for _ in range(n_attributes)])
        # one-hot encode and sort such that value 0 is most frequent overall (value 1 second most frequent and so on)
        self.dataset = torch.nn.functional.one_hot(data.long())[torch.sort((data+0.1).prod(dim=-1)).indices].flatten(start_dim=-2).float().to(device)
        # stores zipf distribution as probality list over all actual items
        normalization = sum([zipf_distribution(val+1) for val in range(len(self.dataset))])
        self.distribution = [zipf_distribution(val+1)/normalization for val in range(len(self.dataset))]
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return self.dataset.__len__()
    
    def __iter__(self):
        '''
        Iterates over dataset in ascening order of frequency rank
        '''
        return self.dataset.__iter__()#self.sample()]
    
    def sampler(self, index=False):
        '''
        Iterator that endlessly samles from the dataset accorrding to zipf distribution
        '''
        while True:
            yield self.sample(index)
            
    def sample(self, index=True):
        i = np.random.choice(len(self.dataset), p=self.distribution)
        if index:
            return i
        else:
            return self.dataset[i]
    
    def plot(self, samplesize=10000):
        counts={str(datum):0 for datum in np.arange(len(self))}#indices of data are sufficient
        for i, datum in enumerate(self.sampler()):
            counts[str(datum)]+=1
            if i==samplesize:
                break

        plt.plot(np.arange(1, len(self)+1), counts.values())
        plt.show()

class Dataset:
    '''
    Note: deprecated by Dataloader
    Class responsible for creating and holding a dataset. Aditionally meant to provide test-train split and 
    all that ML stuff. NOT finetuned for memory, just loads all of it into ram at creation, so don't make it
    too big...
    '''
    def __init__(self, n_attributes, n_values, distribution = "local_values", distribution_param=1.0, data_size_scale=10):
        """
        Creates the dataset at initialiszation and stores it.
        
        
        Parameters
        ----------
            n_attributes : int
                amount of attributes per datum
            n_values : int
                all attributes have the same domain, i.e. arange(n_values)
            distribution : str
                flag for the distribution type, kinda ugly but that way accesible as param of the constructor, also see respective methods
            distribution_param : float
                a parameter of the Zipf-Mandelbrot law
            data_size_scale : float
                see _create_dataset functions for details
        """
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.data_size_scale = data_size_scale
        self.distribution_param = distribution_param
        
        if "local" in distribution:
            '''
            The "same" Values share their frequency rank over different attributes. So e.g. "0" occurs equally often in all Attributes, even though
            it is technically a different concept (e.g. "red" occurs as often as "square", allthough in natural language the distribution over ALL
            words is Zipf distributed.
            '''
            #just in case it is reused, it can be accessed the same way independent of underlying distribution
            self._create_dataset = self._local_attribute_distribution
        
        elif "global" in distribution:
            '''
            This is meant for a future implementation of truly Zipfian distributed "words".
            '''
            raise NotImplementeError
            
        elif "unordered" in distribution:
            '''
            - chatgpt solution, faster but arbitrary order, as well as non-deterministically
            - Riemann Zeta function (???) included in np-zipf
            - dunno if the modulo thing is legit, intuitively it could be sound
            On the bright side: overall dataset param (could be achieved in above implementation
            by simply uniform sampling without replacement for n_dataset times with the same
            non-deterministic drawback)
            '''
            self._create_dataset = self._unordered_global_attribute_distribution
        
        self.dataset = np.array(self._create_dataset(distribution_param=self.distribution_param, data_size_scale=self.data_size_scale))
            
            
    def  _unordered_global_attribute_distribution(self, distribution_param=2, data_size_scale=5):
        """
        Generate a dataset with 3 attributes where the frequency of items follows Zipf's distribution.

        Parameters:
        - num_samples: Number of samples in the dataset.
        - a: Zipf's distribution parameter. (Typically > 1)

        Returns:
        - A pandas DataFrame with the dataset.
        """
        # Generate data based on Zipf's distribution
        attr1 = np.random.zipf(distribution_param, data_size_scale) % self.n_values + 1  # Using modulo to ensure values are between 1-4
        attr2 = np.random.zipf(distribution_param, data_size_scale) % self.n_values + 1
        attr3 = np.random.zipf(distribution_param, data_size_scale) % self.n_values + 1

        # Create a DataFrame
        df = pd.DataFrame({'Attribute1': attr1, 'Attribute2': attr2, 'Attribute3': attr3})

        return df.values
                                                  
                                                  
    def _local_attribute_distribution(self, distribution_param=2, data_size_scale=5):
        '''
        Returns complete dataset as a list of lists.
        Each value is distributed according to Zipf-Mandelbrot law. The absolute frequency of each value 
        relative frequency times data_size_scale. Then all the resulting values are combined wich each value
        from each other attribute. So overall dataset size is:
            n_data = sum_values(floor(Normalized_Zipf-Mandelbrot(value, distribution_param)*data_size_scale))**n_attributes
            or simplified:
            n_data <= dataset_size_scale**n_attributes
        '''
        attribute_values = [[] for _ in range(self.n_attributes)]
        
        # freaking Zipf_mandelbrot is not a distribution, but just proportional to the actual one... So we gotta normalize it with the overall sum so we get relative frequencies
        normalization=sum([zipf_distribution(val+1) for val in range(self.n_values)])
        
        for val in range(self.n_values):
            for i in range(self.n_attributes):
                attribute_values[i] += [val]*int(zipf_distribution(val+1)*data_size_scale/normalization)

        overall_space = np.array(attribute_values[0]).T
        for attribute in attribute_values[1:]:
            overall_space=recursive_cartesian(overall_space, np.array(attribute))
        return overall_space
    
    def plot(self, explicit=False):
        """
        please no one look at efficiency!
        """
        dataset_count={}

        for row in self.dataset:
            try:
                dataset_count[str(row)]+=1
            except KeyError:
                dataset_count[str(row)]=1
                
        # absolute frequencies
        frequency=sorted(dataset_count.values(), reverse=True)

        if explicit:
            plt.plot(sorted(dataset_count.keys(), key=lambda x: dataset_count[x], reverse=True), frequency)
        else:
            plt.plot(np.arange(1,len(frequency)+1), frequency)
            

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = f"n_attributes: {self.n_attributes}\nn_values: {self.n_values}\nn_data: {len(self.dataset)}"
        plt.text(len(frequency)*0.65, max(frequency)*0.95, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
        
        plt.show()
        