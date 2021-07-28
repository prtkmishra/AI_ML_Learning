
class Countvalley:
    """
    An avid hiker keeps meticulous records of their hikes. During the last hike that took exactly  steps, for every step it was noted if it was an uphill, , or a downhill,  step. Hikes always start and end at sea level, and each step up or down represents a  unit change in altitude. We define the following terms:

    A mountain is a sequence of consecutive steps above sea level, starting with a step up from sea level and ending with a step down to sea level.
    A valley is a sequence of consecutive steps below sea level, starting with a step down from sea level and ending with a step up to sea level.
    Given the sequence of up and down steps during a hike, find and print the number of valleys walked through.

    Function Description

    Complete the countingValleys function in the editor below.

    countingValleys has the following parameter(s):

    int steps: the number of steps on the hike
    string path: a string describing the path
    Returns

    int: the number of valleys traversed
    Input Format

    The first line contains an integer , the number of steps in the hike.
    The second line contains a single string , of  characters that describe the path.

    Sample Input

    8
    UDDDUDUU
    Sample Output

    1
    Explanation

    If we represent _ as sea level, a step up as /, and a step down as \, the hike can be drawn as:

    _/\      _
       \    /
        \/\/
    The hiker enters and leaves one valley.
    """
    def __init__(self,steps, path) -> None:
        self.steps = steps
        self.path = path
        self.path = [char for char in self.path.strip()]

    def countingValleys(self):
        sea_level, valleycount  = (0,0)
        vc = []
        for s in self.path:
            if s == "D":
                print("going down")
                sea_level -= 1
            else:
                print("going up")
                sea_level += 1
            if sea_level < 0:
                print("Hiker is in a valley")
                vc.append("valley")
            elif sea_level == 0:
                print("Hiker is at the Sea level")
                vc.append("Sealevel")
            else:
                print("Hiker is on a mountain")
                vc.append("mountain")
        print("Tracking mountains and valleys: ",vc)
        for i in range(len(vc)):
            if vc[i] == "Sealevel":
                if vc[i-1] == "valley":
                    valleycount += 1
        print("Valley Count is: ",valleycount)


        

