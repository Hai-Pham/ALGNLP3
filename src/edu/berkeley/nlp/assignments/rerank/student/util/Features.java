package edu.berkeley.nlp.assignments.rerank.student.util;

import edu.berkeley.nlp.ling.Constituent;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Helper Utility class for extracting features
 */
public class Features {
    public static int getScoreRank(double score) {
        return (int) Math.ceil(Math.exp(score));
    }

    public static int getConstituentLengthLevel(Collection<Constituent<String>> constituents) {
        int size = constituents.size();
        if (size ==1) return 1;
        if (size ==2) return 2;
        if (size ==3) return 3;
        if (size ==4) return 4;
        if (size ==5) return 5;
        if (size <=10) return 6;
        if (size <=20) return 7;
        return 8;
    }

    public static int getSpanLengthGroup(int spanLength) {
        if (spanLength <= 10) return spanLength;
        if (spanLength <=20) return 11;
        return 12;
    }

    public static char getWordShape(String word) {
        char start = word.charAt(0);
        return getCharShape(start);
    }

    private static char getCharShape(char c) {
        if ( (c >= 'a') && (c <= 'z') ) return 'x';
        else if ( (c >= '0') && (c <= '9') ) return 'D'; // digit
        else if ( (c >= 'A') && (c <= 'Z') ) return 'X';
        else return c;
    }

    /**
     * Get 3-gram, with end index (which starts from 0)
     * @param l
     * @param endIdx
     * @return
     */
    public static String get3GramsFromList(List<String> l, int endIdx) {
        return l.get(endIdx - 2) + " " + l.get(endIdx - 1) + " " + l.get(endIdx);
    }


    public static void main(String[] args) {
        String s = "(CEO of 1En34r6no ) said7, \"Too bad, \"";
        System.out.println(getWordShape("\"e"));

        List<String> l = new ArrayList<>();
        l.add("This");
        l.add("is");
        l.add("a");
        l.add("test");
        l.add(".");

        System.out.println(get3GramsFromList(l, 3));

    }
}
