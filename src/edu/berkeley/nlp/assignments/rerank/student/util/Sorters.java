package edu.berkeley.nlp.assignments.rerank.student.util;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Created by Gorilla on 11/10/2016.
 */
public class Sorters {

    public static Integer[] sortIndex(double[] a) {

        Integer[] indices = new Integer[a.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Comparator<Integer> comparator = new Comparator<Integer>() {
            @Override
            public int compare(Integer arg0, Integer arg1) {
                if (a[arg0] > a[arg1]) return 1;
                if (a[arg0] == a[arg1]) return 0;
                return -1;
            }
        };
        Arrays.sort(indices, comparator);

        return indices;
    }

    public static void main(String[] args) {
        double[] a = new double[] {1, 6, 2, 8, 4, 3, 2};

        Integer[] idx = sortIndex(a);
        System.out.println(Arrays.toString(idx));
    }
}

