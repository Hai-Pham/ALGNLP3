package edu.berkeley.nlp.assignments.rerank.student.util;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.SurfaceHeadFinder;
import edu.berkeley.nlp.ling.AnchoredTree;
import edu.berkeley.nlp.ling.Constituent;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.Pair;

import java.util.*;

/**
 * Similar to Naive Feature Extrator but adding a new feature which is the order of score
 */
public class F1FeatureExtractor extends NaiveFeatureExtractor {

    /**
     *
     * @param kbestList
     * @param idx
     *          The index of the tree in the k-best list to extract features for
     * @param featureIndexer
     * @param addFeaturesToIndexer
     *          True if we should add new features to the indexer, false
     *          otherwise. When training, you want to make sure you include all
     *          possible features, but adding features at test time is pointless
     *          (since you won't have learned weights for those features anyway).
     * @return
     */
    @Override
    public int[] extractFeatures(KbestList kbestList, int idx, Indexer<String> featureIndexer, boolean addFeaturesToIndexer) {
        Tree<String> tree = kbestList.getKbestTrees().get(idx);
        List<Integer> feats = new ArrayList<Integer>();

//        double[] scores = kbestList.getScores();
//        addFeature("Score=" + Features.getScoreRank(scores[idx]), feats, featureIndexer, addFeaturesToIndexer);

        // adding this does not improve score
//        Integer[] sortedIdx = Sorters.sortIndex(scores);
//        addFeature("ScoreIdx=" + sortedIdx[idx], feats, featureIndexer, addFeaturesToIndexer);

        // If you just want to iterate over labeled spans, use the constituent list
//        addConstituentFeatures(featureIndexer, addFeaturesToIndexer, tree, feats);


        // Allows you to find heads of spans of preterminals. Use this to fire
        // dependency-based features
        // like those discussed in Charniak and Johnson
        SurfaceHeadFinder shf = new SurfaceHeadFinder();

        // Fires a feature based on the position in the k-best list. This should
        // allow the model to learn that
        // high-up trees
        addFeature("Posn=" + idx, feats, featureIndexer, addFeaturesToIndexer);
//        addFeature("Posn=" + idx, feats, featureIndexer, addFeaturesToIndexer);

        // Converts the tree
        // (see below)
        // You can fire features on parts of speech or words
        List<String> poss = tree.getPreTerminalYield();
        List<String> words = tree.getYield();
        AnchoredTree<String> anchoredTree = AnchoredTree.fromTree(tree);
        extractSubtreeFeatures(featureIndexer, addFeaturesToIndexer, feats, anchoredTree, poss, words);
        // Add your own features here!

        int[] featsArr = new int[feats.size()];
        for (int i = 0; i < feats.size(); i++) {
            featsArr[i] = feats.get(i).intValue();
        }
        return featsArr;
    }


    /**
     * A variance of extractFeatures where we feed a specific tree
     * @param tree
     * @param featureIndexer
     * @param addFeaturesToIndexer
     * @return
     */
    @Override
    public int[] extractFeaturesForSpecificTree(Tree<String> tree, Indexer<String> featureIndexer, boolean addFeaturesToIndexer) {
        // FEATURE COMPUTATION
        List<Integer> feats = new ArrayList<Integer>();

        // If you just want to iterate over labeled spans, use the constituent list
//        addConstituentFeatures(featureIndexer, addFeaturesToIndexer, tree, feats);

        // Allows you to find heads of spans of preterminals. Use this to fire
        // dependency-based features
        // like those discussed in Charniak and Johnson
        SurfaceHeadFinder shf = new SurfaceHeadFinder();


        // Fires a feature based on the position in the k-best list. This should
        // allow the model to learn that
        // high-up trees
        // -1 is for the label
//        addFeature("Posn=-1", feats, featureIndexer, addFeaturesToIndexer);
//        addFeature("Score=0", feats, featureIndexer, addFeaturesToIndexer);

        // Converts the tree
        // (see below)
        // You can fire features on parts of speech or words
        List<String> poss = tree.getPreTerminalYield();
        List<String> words = tree.getYield();
        AnchoredTree<String> anchoredTree = AnchoredTree.fromTree(tree);
        extractSubtreeFeatures(featureIndexer, addFeaturesToIndexer, feats, anchoredTree, poss, words);
        // Add your own features here!

        int[] featsArr = new int[feats.size()];
        for (int i = 0; i < feats.size(); i++) {
            featsArr[i] = feats.get(i).intValue();
        }
        return featsArr;
    }

    /**
     * Helper function for extracting constituent features
     * @param featureIndexer
     * @param addFeaturesToIndexer
     * @param tree
     * @param feats
     */
    private void addConstituentFeatures(Indexer<String> featureIndexer, boolean addFeaturesToIndexer, Tree<String> tree, List<Integer> feats) {
        Collection<Constituent<String>> constituents = tree.toConstituentList();
        addFeature("ConstituentLevel=" + Features.getConstituentLengthLevel(constituents), feats, featureIndexer, addFeaturesToIndexer);
    }

    /**
     * Helper function for extracting features for subtree
     * Will be used twice in kBestlist and in gold tree
     * @param featureIndexer
     * @param addFeaturesToIndexer
     * @param feats
     * @param anchoredTree
     * @param poss
     * @param words
     */
    private void extractSubtreeFeatures3(Indexer<String> featureIndexer, boolean addFeaturesToIndexer, List<Integer> feats,
                                        AnchoredTree<String> anchoredTree, List<String> poss, List<String> words) {
        for (AnchoredTree<String> subtree : anchoredTree.toSubTreeList()) {
            if (!subtree.isPreTerminal() && !subtree.isLeaf()) {
                // Fires a feature based on the identity of a nonterminal rule
                // production. This allows the model to learn features
                // roughly equivalent to those in an unbinarized coarse grammar.
                String rule = "Rule=" + subtree.getLabel() + " ->";
                for (AnchoredTree<String> child : subtree.getChildren()) {
                    rule += " " + child.getLabel();
                }
                addFeature(rule, feats, featureIndexer, addFeaturesToIndexer);

                //====================ADDITIONAL FEATURES==========================
                int startIdx = subtree.getStartIdx();
                int endIdx = subtree.getEndIdx();
                String label = subtree.getLabel();
                // ignore the 1st two levels which covers the whole span of the sentence
                if ( (!label.equals("S")) && (!label.equals("ROOT")) ) {
                    // span length group
                    addFeature("@" + rule +  "^SpanLenGroup=" + Features.getSpanLengthGroup(subtree.getSpanLength()), feats, featureIndexer, addFeaturesToIndexer);

                    // first and last words
                    addFeature("@" + rule +  "^FirstWord=" + words.get(startIdx), feats, featureIndexer, addFeaturesToIndexer);
                    addFeature("@" + rule +  "^LastWord=" + words.get(endIdx - 1), feats, featureIndexer, addFeaturesToIndexer);

                    // first and last POS
                    addFeature("@" + rule +  "^LeftPOS=" + poss.get(startIdx), feats, featureIndexer, addFeaturesToIndexer);
                    addFeature("@" + rule +  "^RightPOS=" + poss.get(endIdx - 1), feats, featureIndexer, addFeaturesToIndexer);

                    // context
                    if (subtree.getStartIdx() >= 1)
                        addFeature("@" + rule +  "^LeftContext=" + words.get(startIdx - 1), feats, featureIndexer, addFeaturesToIndexer);
//                    else
//                        addFeature("@" + rule +  "^LeftContext=BLANK", feats, featureIndexer, addFeaturesToIndexer);
                    if (subtree.getEndIdx() < words.size())
                        addFeature("@" + rule +  "^RightContext=" + words.get(endIdx), feats, featureIndexer, addFeaturesToIndexer);
//                    else
//                        addFeature("@" + rule +  "^RightContext=BLANK", feats, featureIndexer, addFeaturesToIndexer);


                    // get word before split + word after split
                    List<AnchoredTree<String>> subtreeChildren = subtree.getChildren();
                    if (subtreeChildren.size() ==2 ) {
                        AnchoredTree<String> leftChildTree = subtreeChildren.get(0);
                        AnchoredTree<String> rightChildTree = subtreeChildren.get(1);

                        String beforeSplitWord = words.get(leftChildTree.getEndIdx() - 1);
                        String afterSplitWord = words.get(subtreeChildren.get(1).getStartIdx());
                        String leftChildLabel = leftChildTree.getLabel();
                        String rightChildLabel = rightChildTree.getLabel();
                        addFeature(label + "->(" + leftChildLabel + "..." + beforeSplitWord + ")" + rightChildLabel + ")",
                                   feats, featureIndexer, addFeaturesToIndexer);
//                        addFeature("@" + rule + "^SplitPoint=" + beforeSplitWord, feats, featureIndexer, addFeaturesToIndexer);
//                        addFeature("@" + rule + "^SplitPoint=" + afterSplitWord, feats, featureIndexer, addFeaturesToIndexer);
//
//                        //stop here get 86.18 with 1e-1, overfit optimization, 60 loops
//                        // futher add left and right span
//                        addFeature(label + "^LeftSpan=" + Features.getSpanLengthGroup(leftChildTree.getSpanLength()), feats, featureIndexer, addFeaturesToIndexer);
//                        addFeature(label + "^RightSpan=" + Features.getSpanLengthGroup(rightChildTree.getSpanLength()), feats, featureIndexer, addFeaturesToIndexer);
//
//                        // add parent and left dependencies
//                        addFeature(label + "^left=" + leftChildLabel, feats, featureIndexer, addFeaturesToIndexer);
//                        addFeature(label + "^right=" + rightChildLabel, feats, featureIndexer, addFeaturesToIndexer);
                    }

                    // span shape
                    String spanShape = "@" + label + "(";
//                    if (subtree.getSpanLength() <= 5) {
                        for (int i = startIdx; i < endIdx; i++) {
                            spanShape += Features.getWordShape(words.get(i));
                        }
//                    }
//                    else {
//                        spanShape += Features.getWordShape(words.get(startIdx));
//                        spanShape += Features.getWordShape(words.get(startIdx + 1));
//                        spanShape += "AND";
//                        spanShape += Features.getWordShape(words.get(endIdx - 2));
//                        spanShape += Features.getWordShape(words.get(endIdx - 1));
//                    }
                    addFeature(spanShape + ")", feats, featureIndexer, addFeaturesToIndexer);
                }
            }
        }
    }
    private void extractSubtreeFeatures2(Indexer<String> featureIndexer, boolean addFeaturesToIndexer, List<Integer> feats,
                                        AnchoredTree<String> anchoredTree, List<String> poss, List<String> words) {
        for (AnchoredTree<String> subtree : anchoredTree.toSubTreeList()) {
            if (!subtree.isPreTerminal() && !subtree.isLeaf()) {
                // Fires a feature based on the identity of a nonterminal rule
                // production. This allows the model to learn features
                // roughly equivalent to those in an unbinarized coarse grammar.
                String rule = "Rule=" + subtree.getLabel() + " ->";
                for (AnchoredTree<String> child : subtree.getChildren()) {
                    rule += " " + child.getLabel();
                }
                addFeature(rule, feats, featureIndexer, addFeaturesToIndexer);

                //====================ADDITIONAL FEATURES==========================
                int startIdx = subtree.getStartIdx();
                int endIdx = subtree.getEndIdx();
                String label = subtree.getLabel();
                // ignore the 1st two levels which covers the whole span of the sentence
                if ( (!label.equals("S")) && (!label.equals("ROOT")) ) {
                    // span length group
                    addFeature("@" + label +  "^SpanLenGroup=" + Features.getSpanLengthGroup(subtree.getSpanLength()), feats, featureIndexer, addFeaturesToIndexer);

                    // first and last words
                    addFeature("@" + label +  "^FirstWord=" + words.get(startIdx), feats, featureIndexer, addFeaturesToIndexer);
                    addFeature("@" + label +  "^LastWord=" + words.get(endIdx - 1), feats, featureIndexer, addFeaturesToIndexer);

                    // first and last POS
                    addFeature("@" + label +  "^LeftPOS=" + poss.get(startIdx), feats, featureIndexer, addFeaturesToIndexer);
                    addFeature("@" + label +  "^RightPOS=" + poss.get(endIdx - 1), feats, featureIndexer, addFeaturesToIndexer);

                    // context
                    if (subtree.getStartIdx() >= 1)
                        addFeature("@" + label +  "^LeftContext=" + words.get(startIdx - 1), feats, featureIndexer, addFeaturesToIndexer);
                    else
                        addFeature("@" + label +  "^LeftContext=BLANK", feats, featureIndexer, addFeaturesToIndexer);
                    if (subtree.getEndIdx() < words.size())
                        addFeature("@" + label +  "^RightContext=" + words.get(endIdx), feats, featureIndexer, addFeaturesToIndexer);
                    else
                        addFeature("@" + label +  "^RightContext=BLANK", feats, featureIndexer, addFeaturesToIndexer);

                    // get word before split + word after split
                    List<AnchoredTree<String>> subtreeChildren = subtree.getChildren();
                    if (subtreeChildren.size() ==2 ) {
                        String beforeSplit = words.get(subtreeChildren.get(0).getEndIdx() - 1);
                        String afterSplit = words.get(subtreeChildren.get(1).getStartIdx());
                        addFeature("@" + label + "^SplitPoint=" + beforeSplit + ":" + afterSplit, feats, featureIndexer, addFeaturesToIndexer);
                    }

//                    // span shape
//                    String spanShape = "@" + label + "SpanShape=";
//                    if (subtree.getSpanLength() <= 5) {
//                    for (int i = startIdx; i < endIdx; i++) {
//                        spanShape += Features.getWordShape(words.get(i));
//                    }
//                    } else {
//                        spanShape += Features.getWordShape(words.get(startIdx));
//                        spanShape += Features.getWordShape(words.get(startIdx + 1));
//                        spanShape += "N";
//                        spanShape += Features.getWordShape(words.get(endIdx - 2));
//                        spanShape += Features.getWordShape(words.get(endIdx - 1));
//                    }
//                    addFeature(spanShape, feats, featureIndexer, addFeaturesToIndexer);
                }
            }
        }
    }
    private void extractSubtreeFeatures(Indexer<String> featureIndexer, boolean addFeaturesToIndexer, List<Integer> feats,
                                        AnchoredTree<String> anchoredTree, List<String> poss, List<String> words) {
        for (AnchoredTree<String> subtree : anchoredTree.toSubTreeList()) {
            if (!subtree.isPreTerminal() && !subtree.isLeaf()) {
                // Fires a feature based on the identity of a nonterminal rule
                // production. This allows the model to learn features
                // roughly equivalent to those in an unbinarized coarse grammar.
                String rule = "Rule=" + subtree.getLabel() + " ->";
                int count = 0; // counter for 3-grams features
                List<String> threeGrams = new ArrayList<>();
                String label = subtree.getLabel();
                for (AnchoredTree<String> child : subtree.getChildren()) {
                    String childLabel = child.getLabel();
                    rule += " " + childLabel;

                    // process n-grams
                    threeGrams.add(childLabel);
                    if (count >= 2)
                        addFeature(label + " -> " + Features.get3GramsFromList(threeGrams, count), feats, featureIndexer, addFeaturesToIndexer);
                    count++;
                }
                addFeature(rule, feats, featureIndexer, addFeaturesToIndexer);

                //====================ADDITIONAL FEATURES==========================
                int startIdx = subtree.getStartIdx();
                int endIdx = subtree.getEndIdx();

                // ignore the 1st two levels which covers the whole span of the sentence
                if ( (!label.equals("S")) && (!label.equals("ROOT")) ) {
                    // span length group
                    addFeature("@" + rule +  "^SpanLenGroup=" + Features.getSpanLengthGroup(subtree.getSpanLength()), feats, featureIndexer, addFeaturesToIndexer);

                    // first and last words
                    addFeature("@" + rule +  "^FirstWord=" + words.get(startIdx), feats, featureIndexer, addFeaturesToIndexer);
                    addFeature("@" + rule +  "^LastWord=" + words.get(endIdx - 1), feats, featureIndexer, addFeaturesToIndexer);

                    // first and last POS
                    addFeature("@" + rule +  "^LeftPOS=" + poss.get(startIdx), feats, featureIndexer, addFeaturesToIndexer);
                    addFeature("@" + rule +  "^RightPOS=" + poss.get(endIdx - 1), feats, featureIndexer, addFeaturesToIndexer);

                    // context
                    if (subtree.getStartIdx() >= 1)
                        addFeature("@" + rule +  "^LeftContext=" + words.get(startIdx - 1), feats, featureIndexer, addFeaturesToIndexer);
//                    else
//                        addFeature("@" + rule +  "^LeftContext=BLANK", feats, featureIndexer, addFeaturesToIndexer);
                    if (subtree.getEndIdx() < words.size())
                        addFeature("@" + rule +  "^RightContext=" + words.get(endIdx), feats, featureIndexer, addFeaturesToIndexer);
//                    else
//                        addFeature("@" + rule +  "^RightContext=BLANK", feats, featureIndexer, addFeaturesToIndexer);


                    // get word before split + word after split
                    List<AnchoredTree<String>> subtreeChildren = subtree.getChildren();
                    if (subtreeChildren.size() ==2 ) {
                        AnchoredTree<String> leftChildTree = subtreeChildren.get(0);
                        AnchoredTree<String> rightChildTree = subtreeChildren.get(1);

                        String beforeSplitWord = words.get(leftChildTree.getEndIdx() - 1);
                        String afterSplitWord = words.get(subtreeChildren.get(1).getStartIdx());
                        String leftChildLabel = leftChildTree.getLabel();
                        String rightChildLabel = rightChildTree.getLabel();
                        addFeature(label + "->(" + leftChildLabel + "..." + beforeSplitWord + ")" + rightChildLabel + ")",
                                feats, featureIndexer, addFeaturesToIndexer);
                        addFeature("@" + rule + "^SplitPoint=" + beforeSplitWord, feats, featureIndexer, addFeaturesToIndexer);
                        addFeature("@" + rule + "^SplitPoint=" + afterSplitWord, feats, featureIndexer, addFeaturesToIndexer);

//                        //stop here get 86.18 with 1e-1, overfit optimization, 60 loops
//                        // futher add left and right span
                        addFeature(label + "^LeftSpan=" + Features.getSpanLengthGroup(leftChildTree.getSpanLength()), feats, featureIndexer, addFeaturesToIndexer);
                        addFeature(label + "^RightSpan=" + Features.getSpanLengthGroup(rightChildTree.getSpanLength()), feats, featureIndexer, addFeaturesToIndexer);
//
//                        // add parent and left dependencies
                        addFeature(label + "^left=" + leftChildLabel, feats, featureIndexer, addFeaturesToIndexer);
                        addFeature(label + "^right=" + rightChildLabel, feats, featureIndexer, addFeaturesToIndexer);
                    }

                    // span shape
                    String spanShape = "@" + label + "(";
//                    if (subtree.getSpanLength() <= 5) {
                    for (int i = startIdx; i < endIdx; i++) {
                        spanShape += Features.getWordShape(words.get(i));
                    }
//                    }
//                    else {
//                        spanShape += Features.getWordShape(words.get(startIdx));
//                        spanShape += Features.getWordShape(words.get(startIdx + 1));
//                        spanShape += "AND";
//                        spanShape += Features.getWordShape(words.get(endIdx - 2));
//                        spanShape += Features.getWordShape(words.get(endIdx - 1));
//                    }
                    addFeature(spanShape + ")", feats, featureIndexer, addFeaturesToIndexer);
                }
            }
        }
    }

    /**
     * Shortcut method for indexing a feature and adding it to the list of
     * features.
     *
     * @param feat
     * @param feats
     * @param featureIndexer
     * @param addNew
     */
    private void addFeature(String feat, List<Integer> feats, Indexer<String> featureIndexer, boolean addNew) {
        if (addNew || featureIndexer.contains(feat)) {
            feats.add(featureIndexer.addAndGetIndex(feat));
        }
    }



    /**
     * Extract features using SimpleFeaturesExtractor
     * This is a naive way for extraction
     * Will be called by constructor of Awesome Parser
     * Each sample contains; list of k best trees, gold tree, 0-1 loss (size k) of the k trees v.s. the gold one
     */
    @Override
    public List<Datum> simplyExtractFeatures(Iterable<Pair<KbestList, Tree<String>>> kbestListsAndGoldTrees,
                                             Indexer<String> labelIndexer) {
        List<Datum> totalFeaturesWithLabels = new ArrayList<>();

        Iterator<Pair<KbestList, Tree<String>>> iterator = kbestListsAndGoldTrees.iterator();
        while (iterator.hasNext()) {
            Pair<KbestList, Tree<String>> pair = iterator.next();
            KbestList kbestList = pair.getFirst();
            List<Tree<String>> listTreeFromKbestList = kbestList.getKbestTrees();
            Tree<String> goldTree = pair.getSecond();

            List<int[]> kbestListFeatures = new ArrayList<>();
            int[] goldTreeFeatures = null;
            double[] f1Losses = new double[listTreeFromKbestList.size()];

            boolean isGoldTreeInKbestList = false;
            for (int idx = 0; idx < listTreeFromKbestList.size(); idx++) {
                Tree<String> currentTree = listTreeFromKbestList.get(idx);
                int[] treeFeatures = extractFeatures(kbestList, idx, labelIndexer, true);
                kbestListFeatures.add(treeFeatures);
                // if gold tree is one of kbest list
                if (currentTree.toString().equals(goldTree.toString())) {
                    isGoldTreeInKbestList = true;
                    goldTreeFeatures = treeFeatures;
                }
                // get f1 score
                EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> evaluator =
                        new EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>(Collections.singleton("ROOT"), new HashSet<String>());
                double f1 = evaluator.evaluateF1(goldTree, currentTree);
                f1Losses[idx] = (1 - f1);
            }
            // if kbestList does not contain the goldTree, then generate a new set of features for it
            if (!isGoldTreeInKbestList)
                goldTreeFeatures = extractFeaturesForSpecificTree(goldTree, labelIndexer, true);
            // update master list
            totalFeaturesWithLabels.add(new Datum(kbestListFeatures, goldTreeFeatures, f1Losses));
//            debugGoldTreeFeaturesAndKbestListFeatures(kbestListFeatures, goldTreeFeatures);
        }
        return totalFeaturesWithLabels;
    }
}
