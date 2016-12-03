package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.student.util.*;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Created by Gorilla on 11/3/2016.
 */
public class AwesomeParsingRerankerSVM01 implements ParsingReranker {

    private static Indexer<String> labelIndexer = new Indexer<>();

    // TODO: play around with extract features
    NaiveFeatureExtractor featureExtractor = new Zero1FeatureExtractor();
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
        int k = SVMLearner01.svmPredict(weights, kbestList, featureExtractor, labelIndexer);
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
    public AwesomeParsingRerankerSVM01(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
//        debugKbestListAndGoldTreesToConsole(kbestListsAndGoldTrees);
        System.out.println("Extracting features and converting...");
        long start = System.nanoTime();
        convertedTotalFeatures = featureExtractor.simplyExtractFeatures(kbestListsAndGoldTrees, labelIndexer);
        System.out.format("Extracting features takes %d seconds\n", (System.nanoTime() - start)/1000000000);
        System.out.println("Conversion of features is done. Size of label indexer now is : " + labelIndexer.size());

//        debugLabelIndexerToConsole();

        long start2 = System.nanoTime();
        weights = SVMLearner01.svmTrain(convertedTotalFeatures, labelIndexer);
        System.out.format("Training takes %d seconds\n", (System.nanoTime() - start2) / 1000000000);
    }

    //=======================================DEBUG==================================================
    private void debugLabelIndexerToConsole() {
        System.out.println("Content of Label Indexer");
        for (int i = 0; i < labelIndexer.size(); i++) {
            System.out.println(labelIndexer.get(i));
        }
        System.out.println("\n\n");
    }
    public void debugKbestListAndGoldTreesToConsole(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
        Iterator<Pair<KbestList, Tree<String>>> iterator = kbestListsAndGoldTrees.iterator();
        while (iterator.hasNext()) {
            Pair<KbestList, Tree<String>> pair = iterator.next();
            KbestList kbestList = pair.getFirst();
            Tree<String> tree = pair.getSecond();

            List<Tree<String>> kbestTreeList = kbestList.getKbestTrees();
            System.out.println("\n\nThe tree in consideration is: ");
            System.out.println(tree);
            System.out.println("Here is the KBEST LIST");
            for (Tree<String> t : kbestTreeList) {
                System.out.println(t);
            }
            System.out.println();


            double[] scores = kbestList.getScores();
            System.out.println("Scores of kTrees");
            for (double i: scores)
                System.out.print(i + " ");
            System.out.println("\n\n");
        }
    }
    private void debugGoldTreeFeaturesAndKbestListFeatures(List<int[]> kbestListFeatures, int[] goldTreeFeatures) {
        // DEBUG
        System.out.println("Features just extracted:");
        for (int i = 0; i < kbestListFeatures.size(); i++) {
            int[] features = kbestListFeatures.get(i);
            for (int j : features) {
                System.out.print(j + " ");
            }
            System.out.println();
        }
        System.out.println("And the label for those trees is: ");
        for (int k : goldTreeFeatures) {
            System.out.print(k + " ");
        }
        System.out.println();
        // END OF DEBUG
    }
    private void debugAllConvertedFeaturesToConsole(List<Pair<List<IntCounter>, IntCounter>> allConvertedFeatures) {
        int count = 1;
        for (Pair<List<IntCounter>, IntCounter> pair: allConvertedFeatures) {
            System.out.println("Set of KbestList and Gold Tree Features of Order " + count++);
            List<IntCounter> kbestListFeatures = pair.getFirst();
            IntCounter goldTreeFeaturse = pair.getSecond();
            System.out.println("Content of kBestList features");
            for (IntCounter kFeatures: kbestListFeatures)
                printIntCounterToConsole(kFeatures);
            System.out.println("Content of Gold Tree Features");
            printIntCounterToConsole(goldTreeFeaturse);
        }
    }
    private void printIntCounterToConsole(IntCounter counter) {
        Iterable<Map.Entry<Integer, Double>> interable = counter.entries();
        System.out.println("Content of Counter");
        for (Map.Entry<Integer, Double> entry: interable) {
            System.out.format("K=%d V=%.1f\n", entry.getKey(), entry.getValue());
        }
        System.out.println("----------------");
    }
    //=======================================END OF DEBUG===========================================
}
