inputDirectory="../training_data";
outputFile="solution.txt";

dataSets=readAllDatasets(inputDirectory);

generatedRoadSegments={};
for i=1:length(dataSets)
	generatedRoadSegments{end+1,1}=computeAverageTrajectory(dataSets{i});
end

writeSolution(generatedRoadSegments, outputFile);

% FUNCTIONS

% function that computes the road segment from a trajectory set
function averageTrajectory = computeAverageTrajectory(trajectorySet)

	% YOUR CODE SHOULD GO HERE

	% This demo returns the first trajectory in the set
	averageTrajectory = trajectorySet{1};
end

% function reads all the datasets and returns each of them as part of an array
function dataSets = readAllDatasets(inputDirectory)
	files=dir(inputDirectory);
	dataSets =  {};
	for i = 0:length(files)
		fileName=strcat(inputDirectory,'/',int2str(i),'.txt');
		disp(fileName);
		if exist(fileName, 'file') == 2
			 dataSets{end+1,1}=readTrajectoryDataset(fileName);
		end
	end
end

% reads a set of trajectories from a file
function trajectorySet = readTrajectoryDataset(fileName)
	myfile=fopen(fileName);
	tline = fgetl(myfile);
	tlines = cell(0,1);
	
	trajectorySet={};
	while ischar(tline)
		if ~isempty(tline)
			bothNumbers = str2num(tline);
			tlines{end+1,1} = {bothNumbers(1),bothNumbers(2)};
		else
			trajectorySet{end+1,1}=tlines;
			tlines=cell(0,1);
		end
		tline = fgetl(myfile);
	end
	
	trajectorySet{end+1,1}=tlines;
end

% function for writing the result to a file
function writeSolution(generatedRoadSegments, outputFile)
	str="";
	for i = 1:length(generatedRoadSegments)
		segm=generatedRoadSegments{i};
		for j = 1:length(segm)
			str=strcat(str,num2str(segm{j}{1},'%.7f')," ",num2str(segm{j}{2},'%.7f'),'\n');
		end
		str=strcat(str,'\n');
	end
	fileID = fopen(outputFile,'w');
	fprintf(fileID,str);
	fclose(fileID);
end