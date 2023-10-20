

class Dataset:
    '''
    Class responsible for creating and holding a dataset. Aditionally meant to provide test-train split and 
    all that ML stuff. NOT finetuned for memory, just loads all of it into ram at creation, so don't make it
    too big...
    '''
    def __init__(self, n_attributes, n_values, distribution = "local_values", distribution_param=2, data_size_scale=5):
        
        if "local" in distribution:
            '''
            '''
            self.get_dataset = self._local_attribute_distribution
        
        elif "global" in distribution:
            '''
            
            '''
            
        elif "unordered" in distribution:
            '''
            chatgpt solution, faster but somewhat arbitrary
            '''

    def _local_attribute_distribution(n_attributes, n_values, distribution_param=2, data_size_scale=5):
        '''
        placeholder for actual dataset
        val+1 and frequency_rank are the same in this implementation
        '''
        attribute_values = [[] for _ in range(n_attributes)]
        for val in range(n_values):
            for i in range(n_attributes):
                print("bla")
                attribute_values[i] += [val]*int(zipf_distribution(val+1,b=0.5)*data_size_scale)

        overall_space = np.array(attribute_values[0]).T
        for attribute in attribute_values[1:]:
            overall_space=recursive_cartesian(overall_space, np.array(attribute).T)
        return overall_space