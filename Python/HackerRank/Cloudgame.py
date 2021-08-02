
class Cloudgame:
    """
    There is a new mobile game that starts with consecutively numbered clouds. Some of the clouds are thunderheads and others are cumulus. The player can jump on any cumulus cloud having a number that is equal to the number of the current cloud plus  or . The player must avoid the thunderheads. Determine the minimum number of jumps it will take to jump from the starting postion to the last cloud. It is always possible to win the game.

    For each game, you will get an array of clouds numbered  if they are safe or  if they must be avoided.

    Function Description

    Complete the jumpingOnClouds function in the editor below.

    jumpingOnClouds has the following parameter(s):

    int c[n]: an array of binary integers
    Returns

    int: the minimum number of jumps required
    Input Format

    The first line contains an integer , the total number of clouds. The second line contains  space-separated binary integers describing clouds  where .

    Constraints

    Output Format

    Print the minimum number of jumps needed to win the game.

    Sample Input 0

    7
    0 0 1 0 0 1 0
    Sample Output 0

    4
    Sample Input 1

    6
    0 0 0 0 1 0
    Sample Output 1

    3
    Explanation 1:
    The only thundercloud to avoid is . The game can be won in  jumps:
    """
    def __init__(self, c):
        self.c = c

    def jumpingOnClouds(self):
        jumps = 0
        visitedcloud = 0
        for i in range(len(self.c)-1):
            if i < visitedcloud:
                pass
            else:
                if i < len(self.c)-2:
                    if self.c[i] == 1:
                        pass
                    else:
                        if self.c[i] == self.c[i+2]:
                            jumps += 1
                            visitedcloud = i+2
                            print("Jumped from cloud {} to cloud {} ".format(i,i+2))
                        else:
                            if self.c[i] == self.c[i+1]:
                                jumps += 1
                                visitedcloud = i + 1
                                print("Jumped from cloud {} to cloud {} ".format(i,i+1))
                else:
                    if self.c[i] == 1:
                        pass
                    else:
                        if self.c[i] == self.c[i+1]:
                            jumps += 1
                            visitedcloud = i+1
                            print("Jumped from cloud {} to cloud {} ".format(i,i+1))
        print("Total number of jumps in the game: ",jumps)
                

                
