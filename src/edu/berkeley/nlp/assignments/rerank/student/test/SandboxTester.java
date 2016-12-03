package edu.berkeley.nlp.assignments.rerank.student.test;

import edu.berkeley.nlp.ling.Tree;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Gorilla on 11/4/2016.
 */
public class SandboxTester {
    public static void main(String[] args) {
        List<String> list = new ArrayList<>();
        list.add("(ROOT (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))");
        list.add("(ROOT (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays)) (NP (NNP Elianti)) (. .)))");
        list.add("(ROOT (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti) (. .)))))");

        String s = "(ROOT (S (NP (NNP Ms.) (NNP Haag)) (VP (VBZ plays) (NP (NNP Elianti))) (. .)))";
        System.out.println(list.contains(s));


        // loop test
        for (int i = 0; i < 5; ++i)
            System.out.println(i);

    }
}
