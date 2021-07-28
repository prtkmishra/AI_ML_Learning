"""
These are solutions to coding problems from Hacker rank

How to use:
    import the class file
    use it as main with required params
"""

from CountPairsinList import CountPairsinList
from Countvalley import Countvalley

path = "DDUUDDUDUUUD"
steps = 12
#     ar = [10,20, 20, 10, 10, 30, 50, 10, 20]
#     # ar = [6,5,2,3,5,2,2,1,1,5,1,3,3,3,5]
#     n = len(ar)

if __name__ == "__main__":  
    valley = Countvalley(steps, path)
    valley.countingValleys()