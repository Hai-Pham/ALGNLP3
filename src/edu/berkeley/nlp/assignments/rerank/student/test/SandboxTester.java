package edu.berkeley.nlp.assignments.rerank.student.test;

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
    }
}
