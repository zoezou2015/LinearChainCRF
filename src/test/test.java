package test;

import java.io.IOException;

import crf.CRFDriver;
import crf.CRFModel;

public class test {

    
    static String templfile = "data/template";
    static String trainfile = "data/train2.data";
    static String testfile = "data/test2.data";
//    static String modelfile = "data/model.data";

    static CRFModel model;
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        crf_learn();
		crf_test();
		
    }
    public static boolean crf_learn() throws IOException{
        System.out.println("Running CRF learn...");
        CRFDriver driver = new CRFDriver();

//        boolean textmodelfile = false;
//      double freq = 1.0;

        int col_dim = 3;
        int maxIteration = 100;
        double eta = 0.0001;
        double C = 1.0;
        String algorithm = "L-BFGS";
        // crf_learn
        model = driver.crf_learn(templfile, trainfile, col_dim, maxIteration, eta, C, algorithm);
//        model = driver.crf_learn(templfile, trainfile, modelfile, textmodelfile, col_dim,maxIteration, freq, eta, C, algorithm);

        return true;
    }

    /**
     *
     * @return
     * @throws IOException
     */
    public static boolean crf_test() throws IOException{
        System.out.println("\nRunning CRF test...");
        CRFDriver driver = new CRFDriver();

        int col_dim = 3;
        driver.crf_test(templfile, testfile, load_model(), col_dim);

        return true;
    }

    public static CRFModel load_model(){
        return model;
    }

}
