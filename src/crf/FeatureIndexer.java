package crf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class FeatureIndexer{

	class Pair{
		int ID=0;
		int Freq=0;
		public Pair(int id,int freq){
			ID=id;
			Freq=freq;
		}
	}
	private int hiddenSize=0;//the size of hide states 隐藏状态集合的大小，在IndexingHiddenState函数中更新设置
	private int maxFeatureId=0;//the number of feature 特征函数的个数，在IndexingFeatureIndex函数中更新设置
	
	private Map<String, Pair> FeatureIndexMap = new HashMap<String, Pair>();
	private Map<String, Integer> HStateIndexMap = new HashMap<String, Integer>();
	
	public void IndexingHStateIndex(Set<String> hiddenStateTreeSet){
		hiddenSize = hiddenStateTreeSet.size();
		int i=0;
		for(String hidden_state : hiddenStateTreeSet){
			if(hidden_state!=""){
//				System.out.println(hidden_state);
				HStateIndexMap.put(hidden_state, i);
			}
			i++;
		}
	}
	public void IndexingFeatureIndex(TaggerImpl tagger){
		for(ArrayList<String> featureList : tagger.featureStr){
			for(String feature : featureList){
				if(!FeatureIndexMap.containsKey(feature)){
					Pair idFreq=new Pair(maxFeatureId,1);
					FeatureIndexMap.put(feature,idFreq);
					if(feature.startsWith("U")){						
						maxFeatureId+=hiddenSize;	//for unigram feature					
					}else{						
						maxFeatureId+=hiddenSize*hiddenSize; // for bigram feature
					}
				}else{				
					FeatureIndexMap.get(feature).Freq++;
				}
			}
		}
		
	}
	/**
	 * Convert features and hidden states (String) into their index; 
	 * @param tagger
	 */
	public void Register(TaggerImpl tagger){
		for(ArrayList<String> featurelist : tagger.featureStr){
			ArrayList<Integer> fvector=new ArrayList<Integer>();
			for(String feature:featurelist){
				if( FeatureIndexMap.containsKey(feature) ){
					fvector.add(FeatureIndexMap.get(feature).ID);
				}
			}
			tagger.featureList.add(fvector);
		}		
		for(String hiddenstate : tagger.tagStr){			
			tagger.tag.add(HStateIndexMap.get(hiddenstate));
		}
		tagger.tokenSize=tagger.tagStr.size();
		tagger.hiddenSize=hiddenSize;
		
	}
	
	public int getMaxID(){
		return maxFeatureId;
	}
	
	public Map<String, Integer> getFeatureIndexMap(){
		Map<String, Integer> featureIndexMap = new HashMap<String, Integer>();
		Iterator<String> iter = FeatureIndexMap.keySet().iterator();
		while (iter.hasNext()) {
			String feature = iter.next();
			Pair value = FeatureIndexMap.get(feature);
			featureIndexMap.put(feature, value.ID);
		}
		return featureIndexMap;
	}
	
}
