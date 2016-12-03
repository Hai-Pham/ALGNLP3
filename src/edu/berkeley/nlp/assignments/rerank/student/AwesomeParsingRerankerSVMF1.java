package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.student.util.*;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.IntCounter;
import edu.berkeley.nlp.util.Pair;

import java.util.List;

/**
 * Created by Gorilla on 11/3/2016.
 */
public class AwesomeParsingRerankerSVMF1 implements ParsingReranker {

    private static Indexer<String> labelIndexer = new Indexer<>();

    // TODO: play around with extract features
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
        int k = SVMLearnerF1.svmPredict(weights, kbestList, featureExtractor, labelIndexer);
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
    public AwesomeParsingRerankerSVMF1(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees) {
//        debugKbestListAndGoldTreesToConsole(kbestListsAndGoldTrees);
        System.out.println("Extracting features and converting...");
        long start = System.nanoTime();
        convertedTotalFeatures = featureExtractor.simplyExtractFeatures(kbestListsAndGoldTrees, labelIndexer);
        System.out.format("Extracting features takes %d seconds\n", (System.nanoTime() - start)/1000000000);
        System.out.println("Conversion of features is done. Size of label indexer now is : " + labelIndexer.size());

//        debugLabelIndexerToConsole();

        long start2 = System.nanoTime();
        weights = SVMLearnerF1.svmTrain(convertedTotalFeatures, labelIndexer);
        System.out.format("Training takes %d seconds\n", (System.nanoTime() - start2) / 1000000000);
    }

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
}
