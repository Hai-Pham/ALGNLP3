package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.student.util.*;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


public class AwesomeParsingRerankerPerceptron implements ParsingReranker {

    private static Indexer<String> labelIndexer = new Indexer<>();
//    NaiveFeatureExtractor featureExtractor = new NaiveFeatureExtractor();
//    NaiveFeatureExtractor featureExtractor = new Zero1FeatureExtractor();
    NaiveFeatureExtractor featureExtractor = new F1FeatureExtractor();

    List<Datum> convertedTotalFeatures;
    IntCounter weights = null; // weights after tranining

    /**
     * The most important method - will be evaluated by testers
     * @param sentence
     * @param kbestList
     * @return
     */
    @Override
    public Tree<String> getBestParse(List<String> sentence, KbestList kbestList) {
        int k = PerceptronLearner.predictPerceptron(weights, kbestList, featureExtractor, labelIndexer);
        return kbestList.getKbestTrees().get(k);
    }

    //==============================CONSTRUCTOR=======================================
    /**
     * Constructor
     * Get kBestList and Gold Trees (1 tree <=> kbest list>, should help getBestParse()
     * to get best parse tree
     *
     * @param kbestListsAndGoldTrees
     */
    public AwesomeParsingRerankerPerceptron(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
//        debugKbestListAndGoldTreesToConsole(kbestListsAndGoldTrees);
        System.out.println("Extracting features and converting...");
        long start = System.nanoTime();
        convertedTotalFeatures = featureExtractor.simplyExtractFeatures(kbestListsAndGoldTrees, labelIndexer);
        System.out.format("Extracting features takes %d seconds\n", (System.nanoTime() - start)/1000000000);
        System.out.println("Conversion of features is done. Size of label indexer now is : " + labelIndexer.size());

//        debugLabelIndexerToConsole();

        long start2 = System.nanoTime();
        weights = PerceptronLearner.train(convertedTotalFeatures);
        System.out.format("Training takes %d seconds\n", (System.nanoTime() - start2) / 1000000000);
    }


    //==============================CONSTRUCTOR HELPER=======================================
    /**
     * Extract features using SimpleFeaturesExtractor
     * This is a naive way for extraction
     * Will be called by constructor
     */
    private List<Pair<List<int[]>, int[]>> simplyExtractFeatures(Iterable<Pair<KbestList, Tree<String>>>
                                                                         kbestListsAndGoldTrees) {
        Iterator<Pair<KbestList, Tree<String>>> iterator = kbestListsAndGoldTrees.iterator();
        List<Pair<List<int[]>, int[]>> totalFeaturesWithLabels = new ArrayList<>();


        while (iterator.hasNext()) {
            Pair<KbestList, Tree<String>> pair = iterator.next();
            KbestList kbestList = pair.getFirst();
            List<Tree<String>> listTreeFromKbestList = kbestList.getKbestTrees();
            Tree<String> goldTree = pair.getSecond();

            List<int[]> kbestListFeatures = new ArrayList<>();
            int[] goldTreeFeatures = null;
            boolean isGoldTreeInKbestList = false;

            for (Tree<String> t : listTreeFromKbestList) {
                int idx = listTreeFromKbestList.indexOf(t);
                int[] treeFeatures = featureExtractor.extractFeatures(kbestList, idx, labelIndexer, true);
                kbestListFeatures.add(treeFeatures);

                if (t.toString().equals(goldTree.toString())) {
//                    System.out.println("*** GOLD tree in kbest list ***");
                    isGoldTreeInKbestList = true;
                    goldTreeFeatures = treeFeatures;
                }
            }
            // if kbestList does not contain the goldTree, then generate a new set of features
            // for it
            if (!isGoldTreeInKbestList)
                goldTreeFeatures = featureExtractor.extractFeaturesForSpecificTree(goldTree, labelIndexer, true);
            // update master list
            totalFeaturesWithLabels.add(new Pair(kbestListFeatures, goldTreeFeatures));
        }
        return totalFeaturesWithLabels;
    }

    /**
     * Convert the whole totalFeatures to IntCounter to use Dot Product in Machine Learning algorithms
     */
    private List<Pair<List<IntCounter>, IntCounter>> convertAllFeaturesToIntCounter(
            List<Pair<List<int[]>, int[]>> totalFeaturesWithLabels) {
        List<Pair<List<IntCounter>, IntCounter>> convertedFeatures = new ArrayList<>();

        for (Pair<List<int[]>, int[]> pair: totalFeaturesWithLabels) {
            List<int[]> kbestListFeatures = pair.getFirst();
            int[] goldTreeFeaturse = pair.getSecond();
            // update list
            convertedFeatures.add(new Pair(convertListOfFeaturesToIntCounter(kbestListFeatures),
                                           convertFeaturesToIntCounter(goldTreeFeaturse)));
        }
        return convertedFeatures;
    }
    /**
     * Given a list of integer array, convert into IntCounter
     * Using the helper function convertFeaturesToIntCounter()
     */
    private List<IntCounter> convertListOfFeaturesToIntCounter(List<int[]> featureList) {
        List<IntCounter> convertedFeatureList = new ArrayList<>();
        for (int[] features: featureList)
            convertedFeatureList.add(convertFeaturesToIntCounter(features));
        return convertedFeatureList;
    }

    /**
     * Given an integer array, put it into IntCounter to use dot product easily
     * @param features
     * @return
     */
    private IntCounter convertFeaturesToIntCounter(int[] features) {
        IntCounter featureCounter = new IntCounter();
        featureCounter.incrementAll(features, 1);
        return featureCounter;
    }
    //==============================END OF CONSTRUCTOR HELPER=======================================
}
