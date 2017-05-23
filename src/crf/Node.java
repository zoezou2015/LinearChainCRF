package crf;

import java.util.ArrayList;

import org.apache.mahout.math.Vector;

public class Node {
	
	final double LOG2 = 0.69314718055;
	final double MINUS_LOG_EPSILON=50;
	
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
	
	int                  x;
	int                  y;
	double               alpha;
	double               beta;
	double               cost;
	double               bestCost;
	Node                 prev;
	ArrayList<Integer>   fvector=new ArrayList<Integer>();
	ArrayList<CPath>     lpath=new ArrayList<CPath>();
	ArrayList<CPath>     rpath=new ArrayList<CPath>();

	void calcAlpha(){
		alpha = 0.0;
		for(CPath path : lpath){
			alpha = logsumexp(alpha ,path.cost+path.lnode.alpha ,(path == lpath.get(0)));
		}
		alpha+=cost;
//		System.out.println(Math.exp(temp)+" "+ alpha);

	}
	void calcBeta(){
		beta = 0.0;
		for(CPath path : rpath){
			beta = logsumexp(beta ,path.cost+path.rnode.beta ,(path == rpath.get(0)));
		}
		beta+=cost;
//		System.out.println("node beta"+beta);
	}
	void calcGradient(Vector gradients, double Z, int hiddenSize){//gradients应该是引用参数
		double c = Math.exp(alpha + beta - cost - Z);
//		System.out.println("c"+c+" "+alpha+" "+beta+" "+cost+" "+Z);
		for(int f : fvector){//节点的gradient ascent
			gradients.set(f+y, gradients.get(f+y)+c);
		}
		for(CPath path : lpath){
			path.calcGradient(gradients, Z, hiddenSize);
		}
	}
	
	public Node(){
		x =0; y = 0;
	    alpha = beta = cost = 0.0;
	    prev = null;
	    fvector = null;
	    lpath.clear();
	    rpath.clear();
	}
	public Node(int x,int y,ArrayList<Integer> fvector){
		this.x=x;
		this.y=y;
		this.fvector=fvector;
	}
	public void set(int x,int y,ArrayList<Integer> fvector){
		this.x=x;
		this.y=y;
		this.fvector=fvector;
	}
	
}

