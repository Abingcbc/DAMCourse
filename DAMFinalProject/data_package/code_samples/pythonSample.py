# function that computes the road segment from a trajectory set
def computeAverageTrajectory(trajectorySet):

	# YOUR CODE SHOULD GO HERE
	
	# This demo returns the first trajectory in the set
	return trajectorySet[0];

	
# function reads all the datasets and returns each of them as part of an array
def readAllDatasets(inputDirectory):
	dataSets=[];
	import os;
	for i in range(0,len(os.listdir(inputDirectory))):
		fileName=inputDirectory+"/"+str(i)+".txt";
		if(os.path.isfile(fileName)):
			dataSets.append(readTrajectoryDataset(fileName));
	return dataSets;
	
# reads a set of trajectories from a file
def readTrajectoryDataset(fileName):
	s = open(fileName, 'r').read();
	comp=s.split("\n")
	trajectory=[];
	trajectorySet=[];
	for i in range(0,len(comp)):
		comp[i]=comp[i].split(" ");
		if(len(comp[i])==2):
			# to double??
			point={
				"x":float(comp[i][0]),
				"y":float(comp[i][1])
			}
			trajectory.append(point);
		else:
			trajectorySet.append(trajectory);
			trajectory=[];
	
	return trajectorySet;
	
# function for writing the result to a file
def writeSolution(generatedRoadSegments, outputFile):
	string="";
	for i in range(0,len(generatedRoadSegments)):
		segm=generatedRoadSegments[i];
		for j in range(0,len(segm)):
			string+="{:.7f}".format(segm[j]["x"])+" "+"{:.7f}".format(segm[j]["y"])+"\n";
		string+="\n";
		
	f= open(outputFile,"w+");
	f.write(string);
	f.close(); 
	
# MAIN
inputDirectory="../training_data";
outputFile="solution.txt";

dataSets = readAllDatasets(inputDirectory);

generatedRoadSegments=[];
for i in range(0,len(dataSets)):
	generatedRoadSegments.append(computeAverageTrajectory(dataSets[i]));

writeSolution(generatedRoadSegments, outputFile);

	