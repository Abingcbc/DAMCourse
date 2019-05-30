<?php

	/* MAIN */
	
	$inputDirectory="../training_data";
	$outputFile="solution.txt";

	$dataSets = readAllDatasets($inputDirectory);
	
	$generatedRoadSegments=array();
	for($i=0;$i<count($dataSets);$i++){
		$generatedRoadSegments[$i]=computeAverageTrajectory($dataSets[$i]);
	}
	
	writeSolution($generatedRoadSegments, $outputFile);
	
	
	/* FUNCTIONS */
	
	// function that computes the road segment from a trajectory set
	function computeAverageTrajectory($trajectorySet){
	
		// YOUR CODE SHOULD GO HERE
	
		// This demo returns the first trajectory in the set
		return $trajectorySet[0];
	}
	
	// function reads all the datasets and returns each of them as part of an array
	function readAllDatasets($inputDirectory){
		$files = scandir($inputDirectory);
		$dataSets=array();
		for($i=0;$i<count($files);$i++){
			$fileName=$inputDirectory."/".$i.".txt";
			if(file_exists($fileName)){
				$dataSets[$i]=readTrajectoryDataset($fileName);
			}
		}
		return $dataSets;
	}
	
	// reads a set of trajectories from a file
	function readTrajectoryDataset($fileName){
		$myfile = fopen($fileName, "r") or die("Unable to open file!");
		$contents= fread($myfile,filesize($fileName));
		fclose($myfile);
		
		$comp=explode("\n",$contents);
		$trajectorySet=array();
		$trajectory=array();
		for($j=0;$j<count($comp);$j++){
			$vals=explode(" ",$comp[$j]);
			if(count($vals)==2){
				$point=array();
				$point["x"]=floatval($vals[0]);
				$point["y"]=floatval($vals[1]);
				array_push($trajectory,$point);
			}else{
				array_push($trajectorySet,$trajectory);
				$trajectory=array();
			}
		}
		return $trajectorySet;
	}
	
	// function for writing the result to a file
	function writeSolution($generatedRoadSegments, $outputFile){
		$str="";
		for($i=0;$i<count($generatedRoadSegments);$i++){
			$segm=$generatedRoadSegments[$i];
			for($j=0;$j<count($segm);$j++){
				$str.=number_format($segm[$j]["x"],7)." ".number_format($segm[$j]["y"],7)."\n";
			}
			$str.="\n";
		}
		$myfile = fopen($outputFile, "w") or die("Unable to open file!");
		fwrite($myfile, $str);
		fclose($myfile);
	}
	
?>