package edu.berkeley.nlp.assignments.rerank.student.util;

import edu.berkeley.nlp.util.IntCounter;

import java.util.List;
import java.util.Map;

public class IntCounters {
    /**
     * Get difference between 2 vectors
     * Intuitively, this is equivalent to (two - one)
     */
    public static IntCounter getIntCounterVariance(IntCounter one, IntCounter two) {
        IntCounter difference = new IntCounter(one.size()*2);

        // scan thru vector 1
        for (int k1: one.keySet()) {
            double v1 = one.get(k1);
            double v2 = two.get(k1);
            // value 0 does mean nothing, get rid of it
            if (v1 != v2)
                difference.put(k1, v2 - v1);
        }
        // and then vector 2
        for (int k2: two.keySet()) {
            // eliminate the keys already scanned
            // which are the ones that having both v1 and v2
            if (one.get(k2) == 0.0)
                difference.put(k2, two.get(k2));
        }
        return difference;
    }

    /**
     * Update an IntCounter with another IntCounter (including different keys)
     * Intuitively this is equivalent to (one + two)
     * @param one: in our case, one is the difference which (potentially) has more keys
     * @param two: usually the weight vectors
     */
    public static IntCounter addToIntCounterVectorByAnother(IntCounter one, IntCounter two){
        for (int k1: one.keySet()) {
            double v1 = one.get(k1);
            double v2 = two.get(k1);
            two.put(k1, v1 + v2);
        }
        return two;
    }

    /**
     * Extension of addToIntCounterVectorByAnother() by adding a list to a single IntCounter
     * @param list
     * @param two
     * @return
     */
    public static IntCounter addToIntCounterVectorByList(List<IntCounter> list, IntCounter two){
        int listSize = list.size();
        for (IntCounter one: list) {
            for (int k1 : one.keySet()) {
                double v1 = one.get(k1);
                double v2 = two.get(k1);
//                two.put(k1, (v1 + v2)/(double) listSize);
                two.put(k1, (v1 + v2)); // unnormalized
            }
        }
        return two;
    }

    /**
     * Divide an IntCounter by a factor
     * @param counter > 0
     */
    public static IntCounter divideByFactor(IntCounter counter, double factor) {
        for (int k: counter.keySet()) {
            double newVal = counter.get(k) / (double) factor;
            counter.put(k, newVal);
        }

        return counter;
    }



    public static void printIntCounterToConsole(IntCounter counter) {
        Iterable<Map.Entry<Integer, Double>> interable = counter.entries();
        System.out.println("Content of Counter");
        for (Map.Entry<Integer, Double> entry: interable) {
            System.out.format("K=%d V=%.1f\n", entry.getKey(), entry.getValue());
        }
        System.out.println("----------------");
    }


    public static IntCounter convertFeaturesToIntCounter(int[] features) {
        IntCounter featureCounter = new IntCounter();
        for (int k: features) {
            featureCounter.put(k, featureCounter.get(k) + 1);
        }
        //optimization
        features = null;
//        System.gc();

        return featureCounter;
    }


    public static int update(int one, int two) {
        return two + 1;
    }

    public static void main(String[] args) {
        IntCounter i1 = new IntCounter();
        i1.put(1, 1.0);
        i1.put(2, 2.0);

        IntCounter i2 = new IntCounter();
        i2.put(3, 3.0);
        i2.put(4, 4.0);
        i2.put(1, 5.0);

        printIntCounterToConsole(i1);
        printIntCounterToConsole(i2);

        IntCounter i3 = getIntCounterVariance(i1, i2);
        printIntCounterToConsole(i3);

        IntCounter i4 = new IntCounter();
        i4.put(1, 3.0);
        i4 = addToIntCounterVectorByAnother(i3, i4);
        printIntCounterToConsole(i4);
        i4 = addToIntCounterVectorByAnother(i1, i4);
        printIntCounterToConsole(i4);

    }

}
