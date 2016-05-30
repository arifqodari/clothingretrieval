"""
Serialization helper
"""


import cPickle as pickle


def obj_to_str(var):
    """
    Dump the variable into pickled string representation
    """

    return pickle.dumps(var)


def str_to_obj(string):
    """
    Load the string into python object
    """

    return pickle.loads(string)
