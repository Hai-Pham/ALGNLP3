package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.student.util.NaiveFeatureExtractor;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Created by Gorilla on 11/3/2016.
 */
public class AwesomeParsingReranker implements ParsingReranker {

    private static Indexer<String> labelIndexer = new Indexer<>();

    @Override
    public Tree<String> getBestParse(List<String> sentence, KbestList kbestList) {
        return kbestList.getKbestTrees().get(0);
    }


    /**
     * Constructor
     * Get kBestList and Gold Trees (1 tree <=> kbest list>, should help getBestParse()
     * to get best parse tree
     *
     * @param kbestListsAndGoldTrees
     */
    public AwesomeParsingReranker(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
        debugKbestListAndGoldTreesToConsole(kbestListsAndGoldTrees);
        buildLabelIndexer(kbestListsAndGoldTrees);
        List<Pair<List<int[]>, int[]>> totalFeaturesWithLabels = simplyExtractFeatures(kbestListsAndGoldTrees);
        System.out.println("Size of label indexer now is : " + labelIndexer.size());
//        debugLabelIndexerToConsole();

        List<Pair<List<IntCounter>, IntCounter>> convertedTotalFeatures = convertAllFeaturesToIntCounter(totalFeaturesWithLabels);
        debugAllConvertedFeaturesToConsole(convertedTotalFeatures);
    }


    //==============================CONSTRUCTOR HELPER=======================================

    /**
     * Build a label indexer for all the training sentences
     * Called by constructor
     * This is the essential step to extract features from training data
     * (kbest lists + gold trees)
     */
    private void buildLabelIndexer(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
        Iterator<Pair<KbestList, Tree<String>>> iterator = kbestListsAndGoldTrees.iterator();
        while (iterator.hasNext()) {
            Pair<KbestList, Tree<String>> pair = iterator.next();
            Tree<String> tree = pair.getSecond();

            List<String> labelList = tree.getYield();
            for (String label : labelList) {
//                System.out.println(label + " " + labelIndexer.addAndGetIndex(label));
                labelIndexer.add(label);
            }
        }
        System.out.println("Label indexer is done with the size of " + labelIndexer.size());
    }


    /**
     * Extract features using SimpleFeaturesExtractor
     * This is a naive way for extraction
     * Will be called by constructor
     */
    private List<Pair<List<int[]>, int[]>> simplyExtractFeatures(Iterable<Pair<KbestList, Tree<String>>>
                                                                         kbestListsAndGoldTrees) {
        Iterator<Pair<KbestList, Tree<String>>> iterator = kbestListsAndGoldTrees.iterator();
        List<Pair<List<int[]>, int[]>> totalFeaturesWithLabels = new ArrayList<>();

        NaiveFeatureExtractor featureExtractor = new NaiveFeatureExtractor();
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
                    System.out.println("*** GOLD tree in kbest list ***");
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
            debugGoldTreeFeaturesAndKbestListFeatures(kbestListFeatures, goldTreeFeatures);


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


    //=======================================DEBUG==================================================
    /**
     * Debug label indexer
      */
    private void debugLabelIndexerToConsole() {
        System.out.println("Content of Label Indexer");
        for (int i = 0; i < labelIndexer.size(); i++) {
            System.out.println(labelIndexer.get(i));
        }
        System.out.println("\n\n");
    }
    /**
     * Console debuggers
     */
    public void debugKbestListAndGoldTreesToConsole(Iterable<Pair<KbestList, Tree<String>>>
                                                            kbestListsAndGoldTrees) {
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
