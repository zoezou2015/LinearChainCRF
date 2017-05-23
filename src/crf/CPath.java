package crf;



import java.util.ArrayList;

import org.apache.mahout.math.Vector;

public class CPath {

    Node      rnode;
    Node      lnode;
    ArrayList<Integer>   fvector=new ArrayList<Integer>();
    double     cost;

    public CPath(){
        rnode=lnode=null;
        fvector=null;
        cost=0;
    }

    // for CRF
    void calcGradient(Vector gradients, double Z, int hiddenSize){
        double c = Math.exp(lnode.alpha + cost + rnode.beta - Z);
        for(int f : fvector){
        	gradients.set((f+lnode.y*hiddenSize+rnode.y), gradients.get(f+lnode.y*hiddenSize+rnode.y)+c);
        }
    }
    void add(Node _lnode, Node _rnode){
        lnode=_lnode;
        rnode=_rnode;
        lnode.rpath.add(this);
        rnode.lpath.add(this);
    }
    
    
}

