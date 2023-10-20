
def zipf_distribution(freq_rank, a = 1.0, b = 2.7):
    '''
    Default parameters a and b come from Wikipedia/Zipf's_law
    '''
    return 1/(freq_rank+b)**a

def one_hot():
    '''
    transform vectors of size n_attributes to one hot encoded (n_attributes, n_values) arrays
    '''
    pass