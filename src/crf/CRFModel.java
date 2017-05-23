package crf;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

//import org.apache.mahout.classifier.sequencelearning.crf.FeatureIndexer.Pair;
import org.apache.mahout.math.Vector;

public class CRFModel {
	final int version = 100;
	final double const_factor = 1.0;
	
	FeatureTemplate featureTemplate;	
	FeatureExpander featureExpander;
	FeatureIndexer featureIndexer;

	/* the max number of features extracted from corpus，the dimensions of alpha and 
	 * expected are equal to maxid */
	int maxid;
	/* 标注语料库横向的维度,初始化后不能改变 */
	int xsize = 2;

	/*feature weights */
	Vector fweights;
	/* 特征的期望（模型期望与经验期望） */
	Vector expected;
	/* objective function value */
	double obj;
	/* err统计的是当前迭代下，该线程总共预测token错误的个数 */
//	int err;
	/* zeroone统计的是当前迭代下，该线程总共预测sentence错误的个数 */
//	int zeroone;

//	public CRFModel(FeatureTemplate featureTemplate,
//			FeatureExpander featureExpander, FeatureIndexer featureIndexer,
//			Vector fweights, Vector expected, double obj, int err, int zeroone) {
	public CRFModel(FeatureTemplate featureTemplate,
			FeatureExpander featureExpander, FeatureIndexer featureIndexer,
			Vector fweights, Vector expected, double obj) {
		this.featureTemplate = featureTemplate;
		this.featureExpander = featureExpander;
		this.featureIndexer = featureIndexer;
		this.fweights = fweights;
		this.expected = expected;
		this.obj = obj;
//		this.err = err;
//		this.zeroone = zeroone;

		this.maxid = this.featureIndexer.getMaxID();
	}
}