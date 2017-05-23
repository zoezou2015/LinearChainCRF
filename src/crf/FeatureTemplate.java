package crf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class FeatureTemplate {

	ArrayList<String> unigram_templs=new ArrayList<String>();
	ArrayList<String> bigram_templs=new ArrayList<String>();
	
	public static void main(String[] args){
		String url = "data/template";
		try {
			FeatureTemplate ft = new FeatureTemplate(url);
		} catch (Exception e) {
			// TODO: handle exception
		}
		
	}
	
	public FeatureTemplate(String templateUri) throws IOException{
		File file = new File(templateUri);
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(file));
			String templateLine = null;
			int line = 1;
			while ((templateLine = reader.readLine()) != null) {
				if(templateLine.startsWith("U")){
					unigram_templs.add(templateLine);
				}
				if(templateLine.startsWith("B")){
					bigram_templs.add(templateLine);
				}
				//print the template line by line with line number
//				System.out.println("line: "+line  + ": " + templateLine);
				line++;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e1) {
				}
			}
		}
		
	}
}