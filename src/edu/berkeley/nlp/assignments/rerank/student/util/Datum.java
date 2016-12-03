package edu.berkeley.nlp.assignments.rerank.student.util;

import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.List;

/**
 * Datum for LossAugmentedLinearModel
 * Which is the pair of Gold Tree Features + List of Kbest List Trees' Features
 */
public class Datum {

    private List<int[]> candidateTreeFeatures;
    private int[] goldTreeFeatures;
    private boolean[] zeroOneLosses;
    private double[] f1losses;
    /**
     * Take 2 params:
     *      one is the features for a tree in Kbest list
     *      the other is the gold tree feature s
     * @param kbestlistFeatures,
     * @param goldTreeFeatures,
     * @param zeroOneLosses
     */
    public Datum(List<int[]> kbestlistFeatures, int[] goldTreeFeatures, boolean[] zeroOneLosses) {
        this.candidateTreeFeatures = kbestlistFeatures;
        this.goldTreeFeatures = goldTreeFeatures;
        this.zeroOneLosses = zeroOneLosses;
    }

    public Datum(List<int[]> candidateTreeFeatures, int[] goldTreeFeatures, boolean[] zeroOneLosses, double[] f1losses) {
        this.candidateTreeFeatures = candidateTreeFeatures;
        this.goldTreeFeatures = goldTreeFeatures;
        this.zeroOneLosses = zeroOneLosses;
        this.f1losses = f1losses;
    }

    public Datum(List<int[]> candidateTreeFeatures, int[] goldTreeFeatures, double[] f1losses) {
        this.candidateTreeFeatures = candidateTreeFeatures;
        this.goldTreeFeatures = goldTreeFeatures;
        this.f1losses = f1losses;
    }

    public List<int[]> getCandidateTreeFeatures() {
        return candidateTreeFeatures;
    }

    public int[] getGoldTreeFeatures() {
        return goldTreeFeatures;
    }

    public boolean[] getZeroOneLosses() {
        return zeroOneLosses;
    }

    public double[] getF1losses() {
        return f1losses;
    }
}
