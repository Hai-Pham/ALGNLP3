package edu.berkeley.nlp.assignments.rerank.student;

import edu.berkeley.nlp.assignments.rerank.KbestList;
import edu.berkeley.nlp.assignments.rerank.ParsingReranker;
import edu.berkeley.nlp.assignments.rerank.ParsingRerankerFactory;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.Pair;

/**
 * Take the k best list and return a best tree
 */
public class AwesomeParsingRerankerFactory implements ParsingRerankerFactory {

  public ParsingReranker trainParserReranker(Iterable<Pair<KbestList,Tree<String>>>
                                                     kbestListsAndGoldTrees) {
    return new AwesomeParsingReranker(kbestListsAndGoldTrees);
  }
}
