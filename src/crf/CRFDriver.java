package crf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.w3c.dom.css.ElementCSSInlineStyle;

public class CRFDriver{
	
	private static int converge=0;
	/*the max number of features*/
	int maxid;
	/*feature weight*/
	Vector fweights;
	/*gradients*/
	Vector gradients;
	/*objective function value*/
	double obj;
	
//	int err;
//	int zeroone;

	FeatureTemplate featureTemplate; 
	FeatureExpander featureExpander; 
	FeatureIndexer featureIndexer;
	
	ArrayList<TaggerImpl> taggers;
	CRFLBFGS clbfgs;

	public CRFDriver(){
		this.clbfgs = new CRFLBFGS();
	}
	
	/**
	 * Initialize feature weights, expectations, objective function
	 */
	public void initializeCrfDriver(){
		this.maxid = this.featureIndexer.getMaxID();
		
		this.fweights = new DenseVector(this.maxid);
		for(int i=0;i<this.fweights.size();i++){
			this.fweights.set(i, 0.0);
		}
		this.gradients = new DenseVector(this.maxid);// 全局的gradients一致被更新
		for(int i=0;i<this.gradients.size();i++){
			this.gradients.set(i, 0.0);
		}
		this.obj = 0.0;
//		this.err = 0;
//		this.zeroone = 0;
	}
	
	/**
	 * 
	 * @param templfile
	 * @param trainfile
	 * @param modelfile
	 * @param textmodelfile
	 * @param xsize
	 * @param maxitr
	 * @param freq
	 * @param eta
	 * @param C
	 * @param algorithm
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("resource")
//	public CRFModel crf_learn(String templfile, 
//						String trainfile, 
//						String modelfile,
//						boolean textmodelfile, 
//						int xsize, int maxitr, double freq, double eta, double C, String algorithm) throws IOException{
	public CRFModel crf_learn(String templfile, 
			String trainfile, 
			int col_dim, int maxIter, double eta, double C, String algorithm) throws IOException{	
		this.featureTemplate = new FeatureTemplate(templfile); 
		this.featureExpander = new FeatureExpander(this.featureTemplate,col_dim); 
		
		//TaggerImpl:crf算法的计算单元，其对应一个句子
		this.taggers= new ArrayList<TaggerImpl>();
		File trainDataPath = new File(trainfile); //
		BufferedReader reader = new BufferedReader(new FileReader(trainDataPath));
		String line = null;
		ArrayList<String> token_list = new ArrayList<String>();
		
		while ((line = reader.readLine()) != null) {			
			TaggerImpl tagger=new TaggerImpl();	// one tagger for one sentence				
			if( line.trim().equals("") ){
//				System.out.println(tagger.answerStr.size());
				this.featureExpander.expand(token_list,tagger);//add true NER into tagger
//				System.out.println(tagger.answerStr.size());
				this.taggers.add(tagger);
				token_list = new ArrayList<String>();
			}else{
				token_list.add(line);
			}
		}
		
		
		this.featureIndexer = new FeatureIndexer();
		this.featureIndexer.IndexingHStateIndex( this.featureExpander.getHiddenStateSet() );//indexing hidden states
		for (int i = 0; i < this.taggers.size(); i++) {
			TaggerImpl tagger = this.taggers.get(i);
			this.featureIndexer.IndexingFeatureIndex(tagger);//Index feature in tagger 
			this.featureIndexer.Register(tagger);// Convert feature and tag in string into index domain 
		}
		
		this.initializeCrfDriver();
		this.iterateMR(maxIter, eta);
		
//		CRFModel model = new CRFModel(featureTemplate,featureExpander,featureIndexer,fweights,expected,obj,err,zeroone);
		CRFModel model = new CRFModel(featureTemplate,featureExpander,featureIndexer,fweights,gradients,obj);

		return model;
	}

	/**
	 * 
	 * @param templfile
	 * @param testfile
	 * @param model
	 * @param xsize
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("resource")
	public boolean crf_test(String templfile, String testfile,
			CRFModel model, int xsize) throws IOException {
		this.featureExpander = model.featureExpander; //get feature expander from model
		this.featureIndexer = model.featureIndexer; // get feature index from model 
		Set<String> hsSet = this.featureExpander.getHiddenStateSet();
		String hsArray[] = new String[hsSet.size()];
		int id = 0;
		for( String hiddenState:hsSet){
			hsArray[id] = hiddenState;
			id++;
		}
		
		//TaggerImpl:crf算法的计算单元，其对应一个句子
		File testDataPath = new File( testfile );
		BufferedReader reader = new BufferedReader( new FileReader(testDataPath) );
		String line = null;
		ArrayList<String> token_list = new ArrayList<String>();
		ArrayList<String> total_pred = new ArrayList<String>();
		ArrayList<String> total_true = new ArrayList<String>();
		while ( (line = reader.readLine()) != null ) {
//			System.out.println(line);
			TaggerImpl tagger = new TaggerImpl(model.fweights);//为每一个tagger提供其viterbi计算的基础数据:模型的fweights
			if (line.trim().equals("")) {
				this.featureExpander.expand(token_list,tagger );// 特征扩展
				this.featureIndexer.Register(tagger);// 注册tagger	
				tagger.buildLattice();
//				tagger.forwardbackward(); 
				ArrayList<Integer> result = tagger.viterbi();
				
				int tokensNum = token_list.size();
				ArrayList<String> pred_list = new ArrayList<String>(tokensNum);
				ArrayList<String> tag_list = new ArrayList<String>(tokensNum);
				for(int i=0; i<tokensNum; i++){
					String temp = hsArray[result.get(i)];
					pred_list.add(temp);
					String token[]=token_list.get(i).split(" ");
					tag_list.add(token[2]);
//					System.out.println( token_list.get(i) + '\t'+ temp);
//					System.out.println(pred_list.get(i).equals(tag_list.get(i)));
					
				}
				total_true.addAll(tag_list);
				total_pred.addAll(pred_list);
				token_list = new ArrayList<String>();
			} else {
				token_list.add(line);
			}
		}
		evaluate(total_true, total_pred);
		return true;
	}
	
	/**
	 * @return 
	 * 
	 */
	public double run(){
		double C = 10.0;// regularization term parameter
		Vector gradients_current_iteration = new DenseVector(this.maxid);
		for(int i=0;i<gradients_current_iteration.size();i++){
			gradients_current_iteration.set(i, 0.0);
		}
		this.obj = 0.0;
//		this.err = 0;
//		this.zeroone = 0;

		// ---遍历所有的tagger
		for(int i=0; i<taggers.size(); i++){
			TaggerImpl taggerImpl = taggers.get(i);
//			System.out.println("taggerImpl.tokenSize:"+taggerImpl.tokenSize+"   "+"taggerImpl.hiddenSize:"+taggerImpl.hiddenSize);

			taggerImpl.fweights = this.fweights; //taggerImpl.gradient()中利用this.fweights,没有对其做更新
			taggerImpl.gradients = gradients_current_iteration; // update gradients_current_iteration
			
			this.obj += taggerImpl.gradient(); // 梯度计算
			
//			int error_num = taggerImpl.eval();
//			this.err += error_num; 
//			if(error_num!=0){
//				this.zeroone += 1;
//			}
		}

		for(int i=0;i<this.gradients.size();i++){
			this.gradients.set(i, 0.0);
//			this.gradients.set(i, this.gradients.get(i)+gradients_current_iteration.get(i));
			this.gradients.set(i, gradients_current_iteration.get(i));
//			System.out.println(gradients_current_iteration.get(i));
		} 
		
		// L-BFGS算法最优化
		int n = this.maxid;//参数个数 this.fweigths.size()
		double weights [ ]= new double [ n ];//参数向量
		double f ; //目标函数值
		double g [ ] = new double [ n ];//梯度向量

		for(int k=0;k<this.fweights.size();k++){//目标函数值和期望向量用罚函数更新
			this.obj += this.fweights.get(k)*this.fweights.get(k)/(2.0*C);
			this.gradients.set(k,this.gradients.get(k)+this.fweights.get(k)/C);
		}
		/*赋予weights,g,f*/
		for(int i=0; i<this.maxid; i++){
			weights[i]=this.fweights.get(i);
			g[i]=this.gradients.get(i);
		}
		f = this.obj;
		/**
		 * Maximize log-likelihood function f, that is minimize -f
		 * f = s - Z - sum(fweights*fweights)
		 * -f = Z + sum(fweights*fweights) -s
		 */
		this.clbfgs.optimize(n, weights, f, g );//x(参数向量)和f(目标函数值)被更新
		
		/**更新fweights和obj*/
		for(int k=0;k<weights.length;k++){
			this.fweights.set(k, weights[k]);
		}
		this.obj = f;
		return this.obj;//
	}
	
	/**
	 * Run algorithms iteratively and identify if it is converged 
	 * @param numIterations
	 * @param eta
	 * @throws IOException
	 */
	public void iterateMR(int numIterations, double eta) throws IOException{
		System.out.println("Running CRF");
		Double old_obj = new Double(0.0);
		Double obj = new Double(0.0);
		
		int iteration = 1;
		while (iteration <= numIterations) {
//			String jobName = "CRF Iterator running iteration " + iteration;
//			System.out.println(jobName);//Debug
			
			obj = this.run();
			if (isConverged(iteration, numIterations,eta, old_obj, obj)) {
				break;
			}
			old_obj = obj;
			iteration++;
		}
		System.out.println("Encoding finished. Decoding starts...");
	}
	
	
	/**
	 * if the number of iterations is larger than the predefined max iteration numbers,
	 * or the difference between old and current objective function values is smaller than 
	 * some predefined value for 3 times, then the algorithm can be regarded as convergence
	 * @param itr
	 * @param maxitr
	 * @param eta
	 * @param old_obj
	 * @param obj
	 * @return
	 * @throws IOException
	 */
	private boolean isConverged(int itr, int maxitr, double eta, double old_obj, double obj) throws IOException {
//		System.out.println("old_obj:"+old_obj+" "+"obj:"+obj);//调试
		double diff = (itr == 1 ? 1.0 : Math.abs(old_obj - obj) / old_obj);// diff是"相对误差限"

		if (diff < eta) {// /eta=9.99e-005
			converge++;// /如果相对误差限diff小于eta，则converge++
		} else {
			converge = 0;
		}
		// 迭代次数itr大于迭代次数限制，或者converge=3；退出
		if (itr > maxitr  ) {
			System.out.println("itr > maxitr");
			return true;
		} 
		
		if (converge == 3){
			System.out.println("converge == 3");
			return true;
		}
		return false;
	}
	
	private void evaluate(ArrayList<String> tag_true, ArrayList<String> tag_pred){
		if(tag_true.size() != tag_pred.size()){
			System.out.println("The sizes do not match");
			System.exit(1);
		}
		HashMap<String, Integer> tag2count = new HashMap<String, Integer>();
		for (String tag : tag_true){
			if(!tag2count.containsKey(tag)){
				tag2count.put(tag, 0);
				
			} else{
				tag2count.put(tag, tag2count.get(tag)+1);
			}
		}
		
		HashMap<String, Integer> tagPred2count = new HashMap<String, Integer>();
		for (String tag : tag_pred){
			if(!tagPred2count.containsKey(tag)){
				tagPred2count.put(tag, 0);
				
			} else{
				tagPred2count.put(tag, tagPred2count.get(tag)+1);
			}
		}
		
		HashMap<String, Integer> tag2correctPred = new HashMap<String, Integer>();
		for(int i=0; i<tag_true.size();i++){
			String tag1 = tag_true.get(i);
			String tag2 = tag_pred.get(i);
			int pred = tag1.equals(tag2)? 1 : 0;
			if(!tag2correctPred.containsKey(tag1)){
				tag2correctPred.put(tag1, 0);
			} else{
				tag2correctPred.put(tag1, tag2correctPred.get(tag1)+pred);
			}
		}
		
		System.out.println( " "+ '\t' + "precison" + '\t' + "recall" + '\t'+'\t' + "f1-score"+'\t'+"support");
		
		int totalCount = 0;
		double totalPrecison = 0.0;
		double totalRecall = 0.0;
		
		HashMap<String, Double> tag2precison = new HashMap<String, Double>();
		HashMap<String, Double> tag2recall = new HashMap<String, Double>();
		HashMap<String, Double> tag2F1 = new HashMap<String, Double>();
		for(String tag : tag2correctPred.keySet()){
			totalCount += tag2count.get(tag);
			double tempR = tag2correctPred.get(tag)/(tag2count.get(tag)+1E-5);
			double tempP = 0.0;
			double tempF = 0.0;
			if(tagPred2count.containsKey(tag)){
				tempP = tag2correctPred.get(tag)/(tagPred2count.get(tag)+1E-5);
			}
			if(tempR>0)
				totalRecall += tempR*tag2count.get(tag);
			if(tempP>0)
				totalPrecison += tempP*tag2count.get(tag);
			if(tempR*tempP>0)
				tempF = 2*(tempP*tempR)/(tempR+tempP);
			tag2recall.put(tag, tempR);
			tag2precison.put(tag, tempP);
			tag2F1.put(tag, tempF);
			System.out.format("%s\t%8.2f\t%8.2f\t%8.2f\t%8d\n",tag,tag2precison.get(tag),tag2recall.get(tag),tag2F1.get(tag),tag2count.get(tag));
		}
		double R = totalRecall/totalCount;
		double P = totalPrecison/totalCount;
		double F = 2*P*R/(P+R);
		System.out.format("%s\t%2.2f\t%8.2f\t%8.2f\t%8d\n","ave/total:",P,R,F,totalCount);	
	}
	
}
