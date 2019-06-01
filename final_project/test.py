import TRACLUS


def readTrajectoryDataset(fileName):
    s = open(fileName, 'r').read()
    comp = s.split("\n")
    trajectory = []
    trajectorySet = []
    for i in range(0, len(comp)):
        comp[i] = comp[i].split(" ")
        if len(comp[i]) == 2:
            # to double??
            point = {
                "x": float(comp[i][0]),
                "y": float(comp[i][1])
            }
            trajectory.append(point)
        else:
            trajectorySet.append(trajectory)
            trajectory = []

    return trajectorySet

TRACLUS.TRACLUS(readTrajectoryDataset('/Users/cbc/Project/Python/DAMCourse/final_project/data_package/training_data/96.txt'))