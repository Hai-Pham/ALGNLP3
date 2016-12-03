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
public class SVMLearnerF1 extends PrimalSubgradientSVMLearner<Datum> {

    // CONSTANTS for LEARNER
    static final private double REGULARIZATION = 0.01;
    static final private double STEP_SIZE = 0.01; //try 1e0, 1e-1, 1e-2,
    static final private int BATCH_SIZE = 10;
    static final private int NUM_ITERS = 30;

    /**
     * Constructor
     */
    public SVMLearnerF1(int numFeatures) {
        super(STEP_SIZE, REGULARIZATION, numFeatures, BATCH_SIZE);
    }

    //======================== MAIN DRIVER BEGIN ===========================
    /**
     * Given all features, train and return weights
     */
    public static IntCounter svmTrain(List<Datum> convertedTotalFeatures,
                                      Indexer<String> labelIndexer ) {
        int numFeatures = labelIndexer.size();

//        // Straightforward solution
        PrimalSubgradientSVMLearner learner = new PrimalSubgradientSVMLearner(STEP_SIZE, REGULARIZATION, numFeatures, BATCH_SIZE);
        System.out.println("TRAINING NOW USING F1 LOSS...");
        IntCounter weights = learner.train(new IntCounter(numFeatures), new F1LossLinearModel(), convertedTotalFeatures, NUM_ITERS);
        System.out.println("TRAINING DONE! Weights are available for prediction!");

        return weights;

        // Decaying weights
//        PrimalSubgradientSVMLearner learner1 = new PrimalSubgradientSVMLearner(STEP_SIZE, REGULARIZATION, numFeatures, BATCH_SIZE);
//        PrimalSubgradientSVMLearner learner2 = new PrimalSubgradientSVMLearner(STEP_SIZE * 0.1, REGULARIZATION, numFeatures, BATCH_SIZE);
//        PrimalSubgradientSVMLearner learner3 = new PrimalSubgradientSVMLearner(STEP_SIZE * 0.01, REGULARIZATION, numFeatures, BATCH_SIZE);
//        IntCounter weights1 = learner1.train(new IntCounter(numFeatures), new F1LossLinearModel(), convertedTotalFeatures, NUM_ITERS / 3);
//        IntCounter weights2 = learner2.train(weights1, new F1LossLinearModel(), convertedTotalFeatures, NUM_ITERS / 3);
//        IntCounter weights3 = learner3.train(weights2, new F1LossLinearModel(), convertedTotalFeatures, NUM_ITERS / 3);
//        return weights3;
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

//            // optimization - prevent overfitting...
//            for (int k: treeConvertedFeatures.keySet())
//                if ( treeConvertedFeatures.get(k) >= 6)
//                    treeConvertedFeatures.put(k, 0.0);

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