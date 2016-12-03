package edu.berkeley.nlp.assignments.rerank.student.util;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.PrimalSubgradientSVMLearner;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;

import java.util.List;

/**
 * Created by Gorilla on 11/8/2016.
 */
public class SVMLearner01 extends PrimalSubgradientSVMLearner<Datum> {

    // CONSTANTS for LEARNER
    static final private double REGULARIZATION = 0.01;
    static final private double STEP_SIZE = 0.01; //try 1e0, 1e-1, 1e-2,
    static final private int BATCH_SIZE = 10;
    static final private int NUM_ITERS = 30;

    /**
     * Constructor
     */
    public SVMLearner01(int numFeatures) {
        super(STEP_SIZE, REGULARIZATION, numFeatures, BATCH_SIZE);
    }

    //======================== MAIN DRIVER BEGIN ===========================
    /**
     * Given all features, train and return weights
     */
    public static IntCounter svmTrain(List<Datum> convertedTotalFeatures,
                                      Indexer<String> labelIndexer ) {
        int numFeatures = labelIndexer.size();

        // straightforward solution
        PrimalSubgradientSVMLearner learner = new PrimalSubgradientSVMLearner(STEP_SIZE, REGULARIZATION, numFeatures, BATCH_SIZE);
        System.out.println("TRAINING NOW USING ZERO-ONE LOSS...");
        IntCounter weights = learner.train(new IntCounter(numFeatures), new Zero1LossLinearModel(), convertedTotalFeatures, NUM_ITERS);
        System.out.println("TRAINING DONE! Weights are available for prediction!");
        return weights;
    }
    /**
     * Predict the best one given weights ad kbestList
     * @param weights
     * @param kbestList
     * @return
     */
    public static int svmPredict(IntCounter weights, KbestList kbestList,
                                 NaiveFeatureExtractor featureExtractor,
                                 Indexer<String> labelIndexer) {
        List<Tree<String>> listTreeFromKbestList = kbestList.getKbestTrees();

        int bestIndex = -1;
        double maxLoss = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < listTreeFromKbestList.size(); i++) {
            // convert into features space
            int[] treeFeatures = featureExtractor.extractFeatures(kbestList, i, labelIndexer, false);

//             turn into IntCounter to use dot product
            IntCounter treeConvertedFeatures = IntCounters.convertFeaturesToIntCounter(treeFeatures);
            double currentLoss = treeConvertedFeatures.dotProduct(weights);

            // short cut
//            double currentLoss = 0.0;
//            for (int k: treeFeatures)
//                currentLoss += weights.get(k);

            // updating the best index
            if (currentLoss > maxLoss) {
                maxLoss = currentLoss;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    //======================== MAIN DRIVER END ===========================
}