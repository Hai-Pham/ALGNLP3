package edu.berkeley.nlp.assignments.rerank.student.util;

import edu.berkeley.nlp.assignments.rerank.LossAugmentedLinearModel;
import edu.berkeley.nlp.util.IntCounter;

import java.util.List;

// an interface for hooking up a model to an SVM trainer (see comment below)
public class F1LossLinearModel implements LossAugmentedLinearModel<Datum> {

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
        double[] f1Losses = datum.getF1losses();

        double maxLoss = Double.NEGATIVE_INFINITY;
        int idxOfGuess = -1;
        IntCounter lossAugGuessFeatures = new IntCounter();
        for (int i = 0; i < candidateTreeFeatures.size(); i++) {
            // on-the-fly conversion
            IntCounter currentIntCounterFeatures = IntCounters.convertFeaturesToIntCounter(candidateTreeFeatures.get(i));

            double currentF1Loss = f1Losses[i];
            double currentLoss = currentIntCounterFeatures.dotProduct(weights) + currentF1Loss;
            if (currentLoss > maxLoss) {
                maxLoss = currentLoss;
                idxOfGuess = i;
                lossAugGuessFeatures = currentIntCounterFeatures;
            }
        }
        // return the guess tree which yield the highest score
        IntCounter goldIntCounterFeatures = IntCounters.convertFeaturesToIntCounter(goldFeatures);
        double lossOfGuess = f1Losses[idxOfGuess];

//        // optimization - prevent overfitting...
//        for (int k: lossAugGuessFeatures.keySet())
//            if ( lossAugGuessFeatures.get(k) >= 5)
//                lossAugGuessFeatures.put(k, 0.0);


        return new UpdateBundle(goldIntCounterFeatures, lossAugGuessFeatures, lossOfGuess);
    }

}
