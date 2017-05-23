package crf;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Set;
import java.util.TreeSet;

public class FeatureExpander {
	private static int colDim;//标注语料库横向的维度,初始化后不能改 column dimension = 3, the columns of corpus, can not be changed after initialization
	private static int index;//templateLine的字符索引
	static String BOS[] = { "_B-1", "_B-2", "_B-3", "_B-4"};
	static String EOS[] = { "_B+1", "_B+2", "_B+3", "_B+4"};
	
//	private int ysize;//此ysize指的是句子token的数量，他在FeatureStrGenerate()中设置
//	private Set<String> HiddenStateSet=new HashSet<String>();//隐藏状态集合
	private Set<String> HiddenStateSet=new TreeSet<String>();//隐藏状态集合 the set of hidden states 
//	private ArrayList<String>  HiddenStateList=new ArrayList<String>();//隐藏状态列表 the list of hidden states
	
	FeatureTemplate featureTemplate;
	
	public static void main(String[] args){
		String url = "data/template";
		try {
			FeatureTemplate ft = new FeatureTemplate(url);
			FeatureExpander fe = new FeatureExpander(ft, 3);
			
		} catch (Exception e) {
			// TODO: handle exception
		}
		
	}
	
	
	/**
	 * Construct function
	 * @param featureTemplate
	 * 
	 */
	public FeatureExpander(FeatureTemplate featureTemplate,int colDim){
		this.featureTemplate=featureTemplate;//特征模板 feature template
		this.colDim=colDim;
	}

//	/**
//	 * @return the list of hidden states
//	 * 返回句子的隐藏状态序列
//	 */
//	public ArrayList<String> getHiddenStateList(){
//		return HiddenStateList;
//	}
//	
	/**
	 * @return the state predicated tag candidates, where there is no duplication
	 * 返回预测标识集合,用在FeatureIndexMapper
	 */
	public Set<String> getHiddenStateSet(){
		return HiddenStateSet;
	}
	
	/**
	 * 
	 * @param token_list one sentence in corpus
	 * @param tagger
	 * @return
	 */
	public boolean expand(ArrayList<String> token_list,TaggerImpl tagger){
		//the list of list, [[list of one line: token, POS, NER]] for one sentence
		ArrayList<ArrayList<String> > tokenALAL=new ArrayList<ArrayList<String> >();
		
//		int max_xsize=0;int min_xsize=999;
		
		String tokenArr[] = new String[token_list.size()];
		for(int i=0;i<tokenArr.length;i++){
			tokenArr[i] = token_list.get(i);
//			System.out.println("token "  + tokenArr[i]);
		}
		
		for(int i=0;i<tokenArr.length;i++){

			String token[]=tokenArr[i].split(" ");//token[0]:token token[1]:POS token[2]:NER  
			tagger.tagStr.add(token[colDim-1]);// colDim=3, tagStr: the true NER tag
			HiddenStateSet.add(token[colDim-1]); //
			
			ArrayList<String> tokenAL= new ArrayList<String>();
			for(int j=0;j<token.length;j++){
				tokenAL.add(token[j]);
//				System.out.println("state "  +token[j] );

			}
			tokenALAL.add(tokenAL);
		}
		
		//Unigram template to capture unigram features
		for(int i=0;i<tokenALAL.size();i++){
			ArrayList<String> featureAL=new ArrayList<String>(); //the list of feature for one sentence
			for(int j=0;j<featureTemplate.unigram_templs.size();j++){
				StringBuffer feature=new StringBuffer();//
				if(!applyRule(feature,featureTemplate.unigram_templs.get(j),i,tokenALAL)){
					System.out.println("unigram applyRule error");
					return false;
				}
//				System.out.println("state2 "+feature.toString() );
				featureAL.add(feature.toString());
			}
			tagger.featureStr.add(featureAL); 
		}
		//Bigram template to capture bigram features
		for(int i=0;i<tokenALAL.size();i++){
			ArrayList<String> featureAL=new ArrayList<String>();
			for(int j=0;j<featureTemplate.bigram_templs.size();j++){
				StringBuffer feature=new StringBuffer();
				if(!applyRule(feature,featureTemplate.bigram_templs.get(j),i,tokenALAL)){
					System.out.println("bigram applyRule error");
					return false;
				}
				
				featureAL.add(feature.toString());
			}
			tagger.featureStr.add(featureAL); 
		}
		
		return true;
	}

	/**
	 * 删除
	 * 扩展特征
	 * @param sentence
	 * {a1 b1 c1@@a2 b2 c2@@...}的形式
	 * @return
	 */	
//	public ArrayList<String> Expand(String sentence){
//		ArrayList<ArrayList<String> > tokenALAL=new ArrayList<ArrayList<String> >();
//		ArrayList<String> featureAL=new ArrayList<String>();
//		
//		String tokenArr[]=sentence.split("@@");	
//		
//		int max_xsize=0;
//		int min_xsize=999;
//		for(int i=0;i<tokenArr.length;i++){//tokenArr[i]="a1 b1 c1"
//			String token[]=tokenArr[i].split(" ");//token={a1 b1 c1}
//			ArrayList<String> tokenAL= new ArrayList<String>();
//			
//			if(token.length>max_xsize){max_xsize=token.length;}
//			if(token.length<min_xsize){min_xsize=token.length;}
//			for(int j=0;j<token.length;j++){
//				tokenAL.add(token[j]);
//			}
//			tokenALAL.add(tokenAL);
//		}
//		
//		
//		/*验证判断*/
//		if(max_xsize!=min_xsize||xsize>max_xsize){//如果条件成立，舍弃该句子（当然这种判断并不严谨，但暂时这样）
//			System.out.println("ERROR：max_xsize!=min_xsize||xsize>max_xsize");
//			return null;
//		}
//		/*设置与此sentence相关联的数据*/
////		ysize=tokenALAL.size();
//	    for(int i=0;i<tokenALAL.size();i++){
//	    	HiddenStateList.add(tokenALAL.get(i).get(xsize-1));
//	    	HiddenStateSet.add(tokenALAL.get(i).get(xsize-1));
//	    }
//	    
//	    
//	    
//		//Unigram template
//		for(int i=0;i<tokenALAL.size();i++){
//			for(int j=0;j<featureTemplate.unigram_templs.size();j++){
//				StringBuffer feature=new StringBuffer();//
//				if(!applyRule(feature,featureTemplate.unigram_templs.get(j),i,tokenALAL)){
//					System.out.println("unigram applyRule error");
//				}
//				//System.out.println("feature:"+feature);
//				featureAL.add(feature.toString());
//			}
//		}
//		
//		//Bigram template;
//		for(int i=0;i<tokenALAL.size();i++){
//			for(int j=0;j<featureTemplate.bigram_templs.size();j++){
//				StringBuffer feature=new StringBuffer();
//				if(!applyRule(feature,featureTemplate.bigram_templs.get(j),i,tokenALAL)){
//					System.out.println("bigram applyRule error");
//				}
//				//System.out.println("feature:"+feature);
//				featureAL.add(feature.toString());
//			}
//		}
//		
//		return featureAL;
//	}
	
	private boolean applyRule(StringBuffer feature,String tempLine,int pos,ArrayList<ArrayList<String> > sentenceALAL){
		//tempLine="U00:%x[-2,0]"	
		index=0;//index是templine的字符索引
		for(;index<tempLine.length();index++){
			switch (tempLine.charAt(index)){
				default:
					feature.append(tempLine.charAt(index));
					break;
				case '%':
					index++;
					switch (tempLine.charAt(index)){
						case 'x':
							index++;
							String r= getIndex(tempLine, pos, sentenceALAL);//pos, sentenceALAL
							//
							if(r==null){
								return false;
							}
							//System.out.println("r:"+r);
							feature.append(r);
							break;
						default:
							return false;
					}
				break;
				
			}
			
		}
		return true;
	}
	/**
	 * To get the row number
	 * @param tempLine
	 * @param pos
	 * @param sentenceALAL
	 * @return
	 */
	private static String getIndex(String tempLine,int pos,ArrayList<ArrayList<String> > sentenceALAL){
		if(tempLine.charAt(index)!='['){
			return null;
		}
		index++;
		
		int col = 0;
		int row = 0;
		int neg = 1;//neg
		if(tempLine.charAt(index)=='-'){
			neg = -1;
			index++;
		}
		
		for(;index<tempLine.length();index++){
			switch (tempLine.charAt(index)) {
				case '0': case '1': case '2': case '3': case '4':
		        case '5': case '6': case '7': case '8': case '9':
		        	row = 10 * row +(tempLine.charAt(index) - '0');
		        	break;
		        case ',':
		        	index++;
		        	//goto NEXT1;
		        	return NEXT1(tempLine, pos, sentenceALAL,row,col,neg);
		        default: return  null;
			}
		}
		return  null;	
	}
	
	/**
	 * To get column value, and return corresponding feature
	 * @param tempLine
	 * @param pos
	 * @param sentenceALAL
	 * @param row
	 * @param col
	 * @param neg
	 * @return
	 */
	private static  String NEXT1(String tempLine,int pos,ArrayList<ArrayList<String> > sentenceALAL,int row,int col,int neg){
		//NEXT1
		for(;index<tempLine.length();index++){
			switch (tempLine.charAt(index)) {
				case '0': case '1': case '2': case '3': case '4':
		        case '5': case '6': case '7': case '8': case '9':
		        	col = 10 * col +(tempLine.charAt(index) - '0');
		        	break;
		        case ']':
		        {
		        	//NEXT2
		        	row *= neg;
		        	//例：浅层分析中col={0,1,2},xsize=3,xsize=col+1;所以col >=3=(MaxOFcol+1)=(xsize-1+1),条件可以改为col>xsize
		    		if (row < -4 || row > 4 ||col < 0 || col >=colDim ) {
						return null;
					}

		    		int idx = pos + row;
		    		if (idx < 0) {
		    			return BOS[-idx-1];
		    		}
		    		if (idx >= sentenceALAL.size()) {
		    		    return EOS[idx - sentenceALAL.size()];
		    		}
		    		
		    		return sentenceALAL.get(idx).get(col);
		        }
		        	
		        default: return  null;
			}
		}
		return null;
	}
}
