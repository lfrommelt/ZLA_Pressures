
def zipf_distribution(freq_rank, a = 1.0, b = 2.7):
    '''
    Default parameters a and b come from Wikipedia/Zipf's_law
    '''
    return 1/(freq_rank+b)**a


def recursive_cartesian(*arrays):
    '''
    Returns (...((arrays[0] x arrays[1]) x arrays[2]) x ... arrays[-1]) as a two dimensional list
    '''
    new = []
    for value1 in arrays[0]:
        for value2 in arrays[1]:
            try:
                new.append([*value1, value2])
            except TypeError:
                new.append([value1, value2])
    
    if len(arrays)<=2:
        return new
    else:
        return recursive_cartesian(new, *arrays[2:])
    
def one_hot():
    '''
    transform vectors of size n_attributes to one hot encoded (n_attributes, n_values) arrays
    '''
    pass