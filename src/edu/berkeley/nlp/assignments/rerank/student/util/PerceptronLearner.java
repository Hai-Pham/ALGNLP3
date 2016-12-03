package edu.berkeley.nlp.assignments.rerank.student.util;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for Perceptron Learning
 */
public class PerceptronLearner {
    // CONSTANTS for LEARNER
    static private final int NUM_ITERS = 30;
    static private final int BATCH_SIZE = 512;

    public static IntCounter train(List<Datum> convertedTotalFeatures) {
        // PLAY AROUND WITH MANY METHODS OF TRAINING
        return perceptronBatchTrain(convertedTotalFeatures);
//        return perceptronOnlineTrain(convertedTotalFeatures);
    }

    /**
     * Given all features, train and return weights
     */
    public static IntCounter perceptronOnlineTrain(List<Datum> convertedTotalFeatures) {
        System.out.println("TRAINING PERCEPTRON STARTING ...");
        int numSamples = convertedTotalFeatures.size();
        // int weights
        IntCounter weights = new IntCounter();

        for (int i = 0; i < NUM_ITERS; i++) {
            System.out.println("\n\n\nAt loop " + i);
            int numCorrect = 0;
            for (Datum datum: convertedTotalFeatures) {
                // get prediction for each sample
                List<int[]> kbestListFeatures = datum.getCandidateTreeFeatures();
                int[] goldTreeFeatures = datum.getGoldTreeFeatures();
                double[] f1Losses = datum.getF1losses();

//                for (double d: f1Losses)
//                    System.out.println(d);
//                System.out.println("\n\n\n");

                // take the best one in k list and get the loss
                int bestIdx = predictOneSamplePerceptron(kbestListFeatures, weights);

//                if (bestIdx != 0)
//                    System.out.println("Guess idx is NOT ZERO at sample " + convertedTotalFeatures.indexOf(samplePair));

//                double loss = LossFunctions.loss(goldTreeFeatures,
//                        kbestListFeatures.get(bestIdx));

                double loss = f1Losses[bestIdx];
                // if prediction is wrong
                if (loss != 0.0) {
//                    // then update weights
//                    IntCounter goldTreeIntCounterFeatures = IntCounters.convertFeaturesToIntCounter(goldTreeFeatures);
//                    IntCounter currentTreeIntCounterFeatures = IntCounters.convertFeaturesToIntCounter(kbestListFeatures.get(bestIdx));
//                    IntCounter difference = IntCounters.getIntCounterVariance(goldTreeIntCounterFeatures, currentTreeIntCounterFeatures);
////                    System.out.println("Difference");
////                    IntCounters.printIntCounterToConsole(difference);
//                    weights = IntCounters.addToIntCounterVectorByAnother(difference, weights);

                    int[] predictedFeatures = kbestListFeatures.get(bestIdx);
                    for (int k: predictedFeatures)
                        weights.put(k, weights.get(k) - 1);
                    for (int k: goldTreeFeatures)
                        weights.put(k, weights.get(k) + 1);

                } else {
                    // else update number of correct classification
//                    System.out.println("Guessing correctly!");
                    numCorrect++;
                }
                // debug
//                System.out.println("Weights at sample " + convertedTotalFeatures.indexOf(samplePair));
//                IntCounters.printIntCounterToConsole(weights);
            }
            if (numCorrect == numSamples) {
                System.out.println("Converged at loop" + i + ". Exiting now...");
                break;
            }
        }
        System.out.println("TRAINING DONE! Weights are available for prediction!");
        return weights;
    }

    /**
     * Given all features, train and return weights
     * Update weights for every batch size
     */
    public static IntCounter perceptronBatchTrain(List<Datum> convertedTotalFeatures) {
        System.out.println("TRAINING BATCH PERCEPTRON STARTING...");
        int numSamples = convertedTotalFeatures.size();
        int numBatches = (int) Math.ceil(numSamples / (double) BATCH_SIZE);
        // int weights
        IntCounter weights = new IntCounter();
        // debug
        IntCounters.printIntCounterToConsole(weights);


        for (int i = 0; i < NUM_ITERS; i++) {
            System.out.println("\n\n\nAt loop " + i + " of " + NUM_ITERS);
            int totalNumCorrect = 0;
            for (int b = 0; b < numBatches; ++b) {
//                System.out.println("Batch number " + b);
                totalNumCorrect += updateBatch(weights, b, convertedTotalFeatures);
            }
            // terminate when converged
            if (totalNumCorrect == numSamples) {
                System.out.println("Converged at loop " + i + ". Exiting now...\n\n");
                break;
            }
        }
        System.out.println("TRAINING DONE! Weights are available for prediction!");
        return weights;
    }

    /**
     * Helper for weights update for every batch
     * Idea is that (very potentially) updating weights after every sample in a big dataset is unstable
     * So try to update (and normalize) weights after a group of batch
     *
     */
    private static int updateBatch(IntCounter currentWeights, int batchOrder, List<Datum> convertedTotalFeatures) {
        List<Datum> batch = convertedTotalFeatures.subList(batchOrder * BATCH_SIZE,
                Math.min(convertedTotalFeatures.size(), (batchOrder + 1) * BATCH_SIZE));

        int numCorrect = 0;
        List<int[]> guesses = new ArrayList<>(); // record misclassified sample(s)
        List<int[]> golds = new ArrayList<>();
        for (Datum datum: batch) {
            List<int[]> kbestListFeatures = datum.getCandidateTreeFeatures();
            int[] goldTreeFeatures = datum.getGoldTreeFeatures();
            // take the best one in k list and get the loss
            int bestIdx = predictOneSamplePerceptron(kbestListFeatures, currentWeights);
            int[] guessFeatures = kbestListFeatures.get(bestIdx);

            double[] f1Losses = datum.getF1losses();
            double loss = f1Losses[bestIdx];

            if (loss != 0.0) {
                guesses.add(guessFeatures);
                golds.add(goldTreeFeatures);
            } else
                numCorrect++;
        }

        // now after scanning through all samples, update weight just one for the whole batch
//        currentWeights = IntCounters.addToIntCounterVectorByList(varianceOfIncorrectGuesses, currentWeights);
//        IntCounters.divideByFactor(currentWeights, batch.size());
        for (int[] guess: guesses)
            for (int k: guess)
//                currentWeights.put(k, currentWeights.get(k) - 1/(double)batch.size());
                currentWeights.put(k, currentWeights.get(k) - 1);
        for (int[] gold: golds)
            for (int k: gold)
                currentWeights.put(k, currentWeights.get(k) + 1);
//                currentWeights.put(k, currentWeights.get(k) + 1/(double)batch.size());
//        IntCounters.divideByFactor(currentWeights, numCorrect);

        return numCorrect;
    }


    /**
     * Given weigths and kBestList
     * @param kbestListFeatures
     * @param currentWeights
     * @return index of the best tree (which yields the max loss)
     */
    // TODO: move this to IntCounters utility class
//    public static int predictOneSamplePerceptron(List<IntCounter> kbestListFeatures,
//                                                 IntCounter currentWeights){
//        double maxLoss = Double.NEGATIVE_INFINITY;
//        int bestIdx = -1;
//        for (int i = 0; i < kbestListFeatures.size(); i++) {
//            double currentLoss = kbestListFeatures.get(i).dotProduct(currentWeights);
//            if (currentLoss > maxLoss) {
//                bestIdx = i;
//                maxLoss = currentLoss;
//            }
//        }
//        return bestIdx;
//    }

    public static int predictOneSamplePerceptron(List<int[]> kbestListFeatures, IntCounter currentWeights){
        double maxLoss = Double.NEGATIVE_INFINITY;
        int bestIdx = -1;
        for (int i = 0; i < kbestListFeatures.size(); i++) {
//            double currentLoss = kbestListFeatures.get(i).dotProduct(currentWeights);
            double currentLoss = 0.0;
            for (int k: kbestListFeatures.get(i))
                currentLoss += currentWeights.get(k);

            if (currentLoss > maxLoss) {
                bestIdx = i;
                maxLoss = currentLoss;
            }
        }
        return bestIdx;
    }



    /**
     * Prediction given a kbestList
     * @param weights
     * @param kbestList
     * @param featureExtractor
     * @param labelIndexer
     * @return
     */
    public static int predictPerceptron(IntCounter weights, KbestList kbestList,
                                        NaiveFeatureExtractor featureExtractor,
                                        Indexer<String> labelIndexer) {
        List<Tree<String>> listTreeFromKbestList = kbestList.getKbestTrees();

        int bestIndex = -1;
        double maxLoss = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < listTreeFromKbestList.size(); i++) {
            // convert into features space
            int[] treeFeatures = featureExtractor.extractFeatures(kbestList, i, labelIndexer, false);
            // turn into IntCounter to use dot product
//            IntCounter treeConvertedFeatures = convertFeaturesToIntCounter(treeFeatures);
//            double currentLoss = treeConvertedFeatures.dotProduct(weights);
            double currentLoss = 0.0;
            for (int k: treeFeatures)
                currentLoss += weights.get(k);
            // updating the best index
            if (currentLoss > maxLoss) {
                maxLoss = currentLoss;
                bestIndex = i;
            }
        }
        return bestIndex;
    }
    private static IntCounter convertFeaturesToIntCounter(int[] features) {
        IntCounter featureCounter = new IntCounter();
        featureCounter.incrementAll(features, 1);
        return featureCounter;
    }
}
