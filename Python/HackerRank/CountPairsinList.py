#!/bin/python3

import math

class CountPairsinList:
    """
    #
    # Complete the 'sockMerchant' function below.
    #
    # The function is expected to return an INTEGER.
    # The function accepts following parameters:
    #  1. INTEGER n
    #  2. INTEGER_ARRAY ar
    #
    """
    def __init__(self,n,ar):
        self.n = n
        self.ar = ar

    def sockMerchant(self):
        # Write your code here
        _pair = 0
        _sub_list = []
        for i in range(self.n):
            print("Starting element:",self.ar[i])
            if self.ar[i] in _sub_list:
                pass
            else:
                counter = 0
                for j in range(self.n):
                    if self.ar[i] == self.ar[j]:
                        counter += 1
                        print("Index {} is {} the counter is {}".format(j,self.ar[j],counter))
                counter = math.floor(counter/2)
                _pair = _pair + counter
                print("for {} final counter is {}".format(self.ar[i],counter))
            _sub_list.append(self.ar[i])
        print(_pair)

