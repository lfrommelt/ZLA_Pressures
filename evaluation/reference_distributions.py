import numpy as np
import matplotlib.pyplot as plt

class OptimalCoding:
    '''
    Class for getting the lengths of frequency ranked inputs under optimal
    codes (Cover and Thomas, 2006).
    '''
    
    def __init__(self, n_alphabet):
        self.n_alphabet = n_alphabet
        
    def length(self, frequency_rank):
        '''
        Length as function of frequency_rank according to optimal coding
        '''
        #todo: no iteration, use log with base n_alphabet(-1?) somehow
        cum_sum = 0
        i=0
        while True:
            cum_sum += self.n_alphabet**i
            if cum_sum > frequency_rank-1:
                return i+1
            i += 1
            
    def get_sequence(self, max_rank):
        '''
        Returns all the values from 1 to max_rank as list.
        '''
        return [self.length(i) for i in range(1, max_rank)]
        
    def plot(self, max_rank):
        '''
        Plot the length function
        '''
        plt.plot(range(1,max_rank), self.get_sequence(max_rank))
        plt.show()