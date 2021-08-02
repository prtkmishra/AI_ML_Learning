"""
These are solutions to coding problems from Hacker rank

How to use:
    import the class file
    use it as main with required params
    use help() to check the required inputs and description of the problem
"""

from CountPairsinList import CountPairsinList
from Countvalley import Countvalley
from Cloudgame import Cloudgame

# path = "DDUUDDUDUUUD"
# steps = 12
#     ar = [10,20, 20, 10, 10, 30, 50, 10, 20]
#     # ar = [6,5,2,3,5,2,2,1,1,5,1,3,3,3,5]
#     n = len(ar)

c = [0, 0, 0, 1, 0, 0]
if __name__ == "__main__":  
    jumpcount = Cloudgame(c)
    jumpcount.jumpingOnClouds()