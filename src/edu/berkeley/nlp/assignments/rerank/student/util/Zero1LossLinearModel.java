package edu.berkeley.nlp.assignments.rerank.student.util;

import edu.berkeley.nlp.assignments.rerank.LossAugmentedLinearModel;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.List;
import java.util.Map;

// an interface for hooking up a model to an SVM trainer (see comment below)
public class Zero1LossLinearModel implements LossAugmentedLinearModel<Datum> {

    // returns everything an SVM trainer needs to do its thing for a given
    // training datum
    // datum: current training datum... including gold label... you get to define
    // the type T
    // goldFeatures: feature vector of correct label for current training datum
    // lossAugGuessFeatures: feature vector of loss-augmented guess using weights
    // provided for current training datum
    // lossOfGuess: loss of loss-augmented guess compared to gold label for
    // current training datum
    @Override
    public UpdateBundle getLossAugmentedUpdateBundle(Datum datum, IntCounter weights) {
//        IntCounter goldFeatures = datum.getGoldTreeFeatures(); // y*(i)
//        List<IntCounter> candidateTreeFeatures = datum.getCandidateTreeFeatures(); // x(i)
        List<int[]> candidateTreeFeatures = datum.getCandidateTreeFeatures();
        int[] goldFeatures = datum.getGoldTreeFeatures();
        boolean[] zeroOneLosses = datum.getZeroOneLosses();


        double maxLoss = Double.NEGATIVE_INFINITY;
        int idxOfGuess = -1;
        IntCounter lossAugGuessFeatures = new IntCounter();
        for (int i = 0; i < candidateTreeFeatures.size(); i++) {
            // on-the-fly conversion
            IntCounter currentIntCounterFeatures = IntCounters.convertFeaturesToIntCounter(candidateTreeFeatures.get(i));

            double currentZeroOneLoss = zeroOneLosses[i] ? 0.0 : 1.0;
            double currentLoss = currentIntCounterFeatures.dotProduct(weights) + currentZeroOneLoss;
            if (currentLoss > maxLoss) {
                maxLoss = currentLoss;
                idxOfGuess = i;
                lossAugGuessFeatures = currentIntCounterFeatures;
            }
        }
        // return the guess tree which yield the highest score
//        IntCounter lossAugGuessFeatures = candidateTreeFeatures.get(idxOfGuess);
        // now calculate loss from candidate features and gold tree features
        IntCounter goldIntCounterFeatures = IntCounters.convertFeaturesToIntCounter(goldFeatures);
        double lossOfGuess = zeroOneLosses[idxOfGuess] ? 0.0 : 1.0;

        return new UpdateBundle(goldIntCounterFeatures, lossAugGuessFeatures, lossOfGuess);
    }

//    private IntCounter convertFeaturesToIntCounter(int[] features) {
//        IntCounter featureCounter = new IntCounter();
//        featureCounter.incrementAll(features, 1);
//
//        // optimization
//        features = null;
//        System.gc();
//
//        return featureCounter;
//    }



    //    // loss helper function
//    private double zeroOneLoss(IntCounter one, IntCounter two) {
//        if (one.size() != two.size())
//            return 1.0;
//
//        Iterable<Map.Entry<Integer, Double>> interableOne = one.entries();
//        for (Map.Entry<Integer, Double> entry: interableOne) {
//            int k1 = entry.getKey();
//            double v1 = entry.getValue();
//            if (two.get(k1) != v1)
//                return 1.0;
//        }
//        return 0.0;
//    }
}
