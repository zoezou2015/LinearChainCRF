package crf;


import java.util.ArrayList;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;


/*读取TaggerImplWritable,然后反序列化*/
public class TaggerImpl {
	double cost_factor=1.0;
	
	final double LOG2 = 0.69314718055;//ln2=0.69314718055
	final double MINUS_LOG_EPSILON=50;//EPSILON希腊语字母之第五字
	double logsumexp(double x, double y, boolean flg) {
		if (flg) return y;  // init mode
		double vmin = Math.min(x, y);
		double vmax = Math.max(x, y);
		if (vmax > vmin + MINUS_LOG_EPSILON) {
			return vmax;
		} else {
			return vmax + Math.log(Math.exp(vmin - vmax) + 1.0);
		}
	}

	/** TaggerImplWritable中需要存储的 */
	// feature in string format
	ArrayList<ArrayList<String> > featureStr=new ArrayList<ArrayList<String> >();
	// tag in string format
	ArrayList<String> tagStr=new ArrayList<String>();
	/**
	 * 特征索引矩阵featureList的元素都是fvector，
	 * 前tokenSize个是状态特征的，后tokenSize个是转移特征的；因此featureList是（tokenSize*2）*（fvector.size()）维,按行遍历
	 */
	ArrayList<ArrayList<Integer> > featureList=new ArrayList<ArrayList<Integer> >();
	// the corresponding tag index for tokens
	ArrayList<Integer> tag=new ArrayList<Integer>();
	// token size in one sentence 句子token的数量
	int tokenSize=0;
	// 预测标记(隐藏状态)集合的大小,不是句子token扩展出来的特征的数量
	int hiddenSize=0;
	
	/** TaggerImplWritable中不需要存储的，是计算得来的*/
	// 罚函数
	ArrayList<ArrayList<Double> > penalty=new ArrayList<ArrayList<Double> >();
	// 预测结果
	ArrayList<Integer> predTag =new ArrayList<Integer>();
	int nbest;
	double cost;
	//normalization term
	double Z;
	
	/**
	 * lattic网格
	 * nodeList共有（tokenSize）*（hiddenSize）个节点，可以画一个二维的坐标轴，坐标（x,y）与索引index的对应关系是index=x*hiddenSize+y
	 * pathList共有（tokenSize-1）*(hiddenSize*hiddenSize)个路径，可以尝试在nodeList中链接所有的节点，循环中（cur:2->tokenSize，j,k）对应的索引index=(cur-1)*hiddenSize*hiddenSize+j*hiddenSize+k
	 */
	ArrayList<Node> nodeList;
	ArrayList<CPath> pathList;
	
	/**alpha与expected两者，只是在这里定义，其引用的是mapper中为两者分配alpha与expected*/
	//  feature weights
	Vector fweights=new DenseVector();
	// gradients 
	Vector gradients=new DenseVector();
	
	
	/**
	 * 
	 * @param featureStr
	 * @param tagStr
	 * @param featureList
	 * @param tag
	 * @param tokenSize
	 * @param hiddenSize
	 */
	public TaggerImpl(ArrayList<ArrayList<String> > featureStr,ArrayList<String> tagStr,
			ArrayList<ArrayList<Integer> > featureList,ArrayList<Integer> tag,int tokenSize,int hiddenSize){
		this.featureStr=featureStr;
		this.tagStr=tagStr;
		this.featureList=featureList;
		this.tag=tag;
		this.tokenSize=tokenSize;
		this.hiddenSize=hiddenSize;
	}
	public TaggerImpl(Vector fweights){
		this.fweights = fweights;
	}
	public TaggerImpl(){}
	
	/**
	 * buildLattice
	 * 创建网格（节点与边）
	 */
	public void buildLattice() {
		LatticAllocate();//为nodeList和pathList初始化和分配空间
		
		if(featureList.isEmpty()){
			return ;
		}
		//构建网格（节点和边）lattic
		int fid=0;
		for(int cur=0;cur<tokenSize;cur++){//节点
			ArrayList<Integer> fvector=featureList.get(fid++);
			for(int j=0;j<hiddenSize;j++){
				Lattic(cur,j).set(cur,j,fvector);
			}
		}
		for(int cur=1;cur<tokenSize;cur++){//路径
			ArrayList<Integer> fvector=featureList.get(fid++);
			for(int j=0;j<hiddenSize;j++){
				for(int k=0;k<hiddenSize;k++){
					Lattic(cur,j,k).add(Lattic(cur-1,j), Lattic(cur,k));
					Lattic(cur,j,k).fvector=fvector;
				}
			}
		}

		//计算节点和路径的cost
		for(int cur=0;cur<tokenSize;cur++){
			for(int j=0;j<hiddenSize;j++){
				calcCost(Lattic(cur,j));//node cost
				for(int pindex=0;pindex<Lattic(cur,j).lpath.size();pindex++){
					calcCost(Lattic(cur,j).lpath.get(pindex));//path cost
				}
			}
		}

	}
	
	/**
	 * LatticAllocate
	 * Allocate space for nodeList, pathList and predTag
	 * 为nodeList，pathList和predTag分配初始工作空间
	 */
	private void LatticAllocate(){
		// ArrayList初始化即分配空间，提高效率
		int nodeNum=tokenSize*hiddenSize;
		nodeList=new ArrayList<Node>(nodeNum);
		for(int i=0;i<nodeNum;i++){
			nodeList.add(new Node());
		}
		
		int pathNum=(tokenSize-1)*hiddenSize*hiddenSize;
		pathList=new ArrayList<CPath>(pathNum);
		for(int i=0;i<pathNum;i++){
			pathList.add(new CPath());
		}
		
//		for(int i=0;i<tokenSize;i++){
//			predTag.add(0);
//		}
	}
	
	/**
	 * 计算节点的cost
	 * @param n
	 */
	private void calcCost(Node n) {
		double c=0;
		ArrayList<Integer> fvector=n.fvector;
		/**
		 * sum{feature_weights[k]*feature[k]} for unigram features
		 */
		for(int f : fvector){
			c +=fweights.get(f+n.y);//f:feature index, n.y:node hidden state index 
		}
		n.cost = c;
//		System.out.println("node alpha "+c);

	}
	/**
	 * 计算路径的cost
	 * @param p
	 */
	private void calcCost(CPath p) {
		double c=0;
		ArrayList<Integer> fvector=p.fvector;
		/**
		 * sum{feature_weights[k]*feature[k]} for bigram features
		 */
		for(int f : fvector){
			c +=fweights.get(f+p.lnode.y*hiddenSize+p.rnode.y);//for each size, there are hiddenSize*hiddenSize possible feature
		}
		p.cost = c;
//		System.out.println("path alpha "+c);
	}
	
	/**
	 * 前向后向算法
	 */
	public void forwardbackward() {
		if(featureList.isEmpty()){
			return;
		}
		
		for(int i=0;i<tokenSize;i++){//计算节点的alpha值
			for(int j=0;j<hiddenSize;j++){
				Lattic(i,j).calcAlpha();
			}
		}
		
		for(int i=tokenSize-1;i>=0;i--){//计算节点的beta值，需反向计算
			for(int j=0;j<hiddenSize;j++){
				Lattic(i,j).calcBeta();
			}
		}               
		
		Z=0.0;
		for (int j = 0; j < hiddenSize; ++j){//根据节点的beta值，计算归一化因子Z
			Z = logsumexp(Z, Lattic(tokenSize-1,j).alpha, j == 0);
		}
		
//		VerifyFB();	
	}
	
	/**
	 * viterbi算法
	 */
	public ArrayList<Integer> viterbi() {
		
		//find best path and previous node for each node
		for(int i=0;i<tokenSize;i++){//token的遍历
			for(int j=0;j<hiddenSize;j++){//隐藏状态序列的遍历
				double bestc = -1e37;
				Node best = null;
				for(CPath path : Lattic(i,j).lpath){
					double cost = path.lnode.bestCost+path.cost+Lattic(i,j).cost;
					if (cost > bestc) {
				          bestc = cost;
				          best  = path.lnode;
					}
				}
				Lattic(i,j).prev      =best;//最优路径上当前节点的前驱
				Lattic(i,j).bestCost  =(best!=null) ? bestc :Lattic(i,j).cost;//最优路径上当前节点的bestCost
			}
		}
		
		//backtrace 
		double bestc = -1e37;
		Node best = null;
		int s = tokenSize-1;
		for(int j=0;j<hiddenSize;j++){
			if(bestc < Lattic(s,j).bestCost){
				best=Lattic(s,j);
				bestc=Lattic(s,j).bestCost;
			}
		}

		predTag=new ArrayList<Integer>();
		for(int i=0;i<tokenSize;i++){
			predTag.add(0);
		}
		for(Node n=best;n!=null;n=n.prev){
			predTag.set(n.x, n.y);//
		}
//		cost=-Lattic(tokenSize-1,predTag.get(tokenSize-1)).bestCost;//
		
		return predTag;
	}
	
	/**
	 * eval()
	 * @return
	 */
	public int eval() {
		// TODO Auto-generated method stub
		int err = 0;
		for (int i = 0; i < tokenSize; ++i) {
			if(tag.get(i)!=predTag.get(i)){
				++err;
			}
		}
		return err;
	}
	
	/**
	 * 计算梯度
	 * @return
	 * 返回obj
	 */
	public double gradient() {
		if(featureList.isEmpty()){
			return 0.0;
		}

		buildLattice();
		forwardbackward();
		
		double s = 0.0;
		for(int i=0;i<tokenSize;i++){
			for(int j=0;j<hiddenSize;j++){
				Lattic(i,j).calcGradient(gradients, Z, hiddenSize);//【节点node和边path的gradient】
			}
		}
		
		for(int i=0;i<tokenSize;i++){
			Node selectedNode=Lattic(i,tag.get(i));//selectedNode
			ArrayList<Integer> fvector=selectedNode.fvector;

			for(int f : fvector){
				int index=f+tag.get(i);//expected的索引
				gradients.set(index, gradients.get(index)-1);
			}

			s+=selectedNode.cost;
			
			ArrayList<CPath> pathAL=selectedNode.lpath;
			for(int j=0;j<pathAL.size();j++){
				Node lnode=pathAL.get(j).lnode;//该路径pathAL.get(j)的左边节点
				Node rnode=pathAL.get(j).rnode;//该路径pathAL.get(j)的右边节点
				if(lnode.y==tag.get(lnode.x)){//pathAL.get(j).lnode的x和y(条件是y=answer[x])
					ArrayList<Integer> pvector=pathAL.get(j).fvector;//路径的fvector
					for(int f : pvector){
						int index=f+lnode.y*hiddenSize+rnode.y;//expected的索引
						gradients.set(index, gradients.get(index)-1);
					}
					s+=pathAL.get(j).cost;
					break;
				}
			}
			
		}

		return Z - s ;
		
	}
	
	/**
	 * 
	 * @param x
	 * 横轴坐标：0->tokenSize
	 * @param y
	 * 纵轴坐标：0->hiddenSize
	 * @return
	 * 返回Node的引用
	 */
	private Node Lattic(int x,int y){
		return nodeList.get(x*hiddenSize+y);
	}
	/**
	 * 
	 * @param cur
	 * cur:1->tokenSize
	 * @param j
	 * j:0->hiddenSize
	 * @param k
	 * y:0->hiddenSize
	 * @return
	 */
	private CPath Lattic(int cur,int j,int k){
		return pathList.get((cur-1)*hiddenSize*hiddenSize+j*hiddenSize+k);
	}
	
	private void VerifyFB(){

		double alphaCount = 0.0;
		double betaCount = 0.0;
		ArrayList<Double> countList = new ArrayList<>();
		for (int j = 0; j < hiddenSize; ++j){//根据节点的beta值，计算归一化因子Z
			alphaCount = logsumexp(alphaCount, Lattic(tokenSize-1,j).alpha, j == 0);
			betaCount = logsumexp(betaCount, Lattic(0,j).beta, j == 0);
		}
		System.out.println("alpha Final: "+alphaCount);
		System.out.println("beta 0 : "+ betaCount);
		for(int i=0; i<tokenSize;i++){
			double middleCount = 0.0;		
			for(int j=0; j<hiddenSize; j++){
			middleCount = logsumexp(middleCount, Lattic(i,j).alpha+Lattic(i,j).beta, j == 0);				
			}
			countList.add(middleCount);
		}
		for(int i =0; i<tokenSize; i++){
			System.out.println("position "+i+" value "+countList.get(i));
		}
		
		System.out.println("=========");
//		System.out.println("alpha: "+alphaCount);
//		System.out.println("beta: "+betaCount);
//		double forCount = 0.0;
//		double backCount = 0.0;
//		double t = 0.0;
//		for(int i=0; i<hiddenSize; i++){
//			for(CPath path : Lattic(1,i).lpath){
//				t = logsumexp(t, path.cost, (path == Lattic(1,i).lpath.get(0)));
//			}
//			t+=Lattic(1,i).cost;
//			forCount = logsumexp(forCount, Math.log(Math.exp(Lattic(1,i).alpha)+Math.log(Math.exp(t))+Math.exp(Lattic(1,i).beta)), i == 0);
////			backCount = logsumexp(backCount, Math.log(Math.exp(Lattic(0,i).alpha)*Math.exp(Lattic(0,i).beta)), i == 0);
//		}

//		System.out.println("alpha2: "+alphaCount);
//		System.out.println("beta: "+betaCount);
//		if((alphaCount-betaCount)<1E-5)
//			System.out.println("Verify true");
//		else
//			System.out.println("Verify false");
	}
	
}
