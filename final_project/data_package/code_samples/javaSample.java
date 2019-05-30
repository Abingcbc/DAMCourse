import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class javaSample {
	static String inputDirectory="../training_data";
	static String outputFile="solution.txt";
	
	public static void main(String[] args) throws IOException {

		ArrayList<ArrayList<Trajectory>> dataSets = readAllDatasets(inputDirectory);
		
		ArrayList<Trajectory> generatedRoadSegments=new ArrayList<Trajectory>();
		for(int i=0;i<dataSets.size();i++){
			generatedRoadSegments.add(computeAverageTrajectory(dataSets.get(i)));
		}
		
		writeSolution(generatedRoadSegments, outputFile);
		
	}
	
	// function that computes the road segment from a trajectory set
	public static Trajectory computeAverageTrajectory(ArrayList<Trajectory> trajectorySet){
	
		// YOUR CODE SHOULD GO HERE
		
		// This demo returns the first trajectory in the set
		return trajectorySet.get(0);
	}
	
	// function reads all the datasets and returns each of them as part of an array
	public static ArrayList<ArrayList<Trajectory>> readAllDatasets(String inputDirectory) throws IOException{
		File directory = new File(inputDirectory);

		String [] directoryContents = directory.list();

		ArrayList<ArrayList<Trajectory>>dataSets=new ArrayList<ArrayList<Trajectory>>();
		for(int i=0;i<directoryContents.length;i++){
			String fileName=inputDirectory+"/"+i+".txt";
			
			File file = new File(fileName);
			if(file.exists()){
				dataSets.add(readTrajectoryDataset(fileName));
			}
		}
		return dataSets;
	}
	
	// reads a set of trajectories from a file
	public static ArrayList<Trajectory> readTrajectoryDataset(String fileName) throws IOException{
		BufferedReader br=new BufferedReader(new FileReader(fileName));
		String line;
		javaSample main=new javaSample();
		Trajectory trajectory=main.new Trajectory();
		ArrayList<Trajectory> trajectorySet=new ArrayList<Trajectory>();
	
		while((line=br.readLine())!=null){
			String[] vals = line.split("\\s+");
			if(vals.length==2){
				Point p=main.new Point(vals[0],vals[1]);
				trajectory.add(p);
			}else{
				trajectorySet.add(trajectory);
				trajectory=main.new Trajectory();
			}
		}
		br.close();
		return trajectorySet;
	}
	
	// function for writing the result to a file
	public static void writeSolution(ArrayList<Trajectory> generatedRoadSegments, String outputFile) throws IOException{
		String str="";
		for(int i=0;i<generatedRoadSegments.size();i++){
			Trajectory segm=generatedRoadSegments.get(i);
			for(int j=0;j<segm.size();j++){
				str+=String.format("%.7f", segm.get(j).x)+" "+String.format("%.7f", segm.get(j).y)+"\n";
			}
			str+="\n";
		}
		
		BufferedWriter bw=new BufferedWriter(new FileWriter(outputFile));
		bw.write(str);
		bw.close();
	}
	

	// Helper classes
	
	class Trajectory{
		ArrayList<Point> points;
		public Trajectory(){
			points=new ArrayList<Point>();
		}
		public void add(Point p){
			points.add(p);
		}
		public Point get(int index){
			return points.get(index);
		}
		public int size(){
			return points.size();
		}
	}
	
	class Point{
		public double x;
		public double y;
		public Point(double x, double y){
			this.x=x;
			this.y=y;
		}
		public Point(String x, String y){
			this.x=Double.parseDouble(x);
			this.y=Double.parseDouble(y);
		}
	}
}
