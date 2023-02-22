package Evaluator;

import other.context.Context;

import java.util.Random;

/**
 * Generates a leaf evaluator that returns a random value
 */
public class RandomLeafEvaluator extends GameStateEvaluator {

    //-------------------------------------------------------------------------

    /**
     * Randomgenerator to generate pseudorandom numbers
     */
    protected Random rand;

    //-------------------------------------------------------------------------

    /**
     * Constructor requiring and using no inputs
     */
    public RandomLeafEvaluator() {
        // Creating random generator
        this.rand = new Random();
    }

    /**
     * Generates a pseudorandom number (ignoring the context or maximising player)
     *
     * @param context          Ludii's context
     * @param maximisingPlayer The maximising player
     * @return A pseudorandom number
     */
    public float evaluate(Context context, int maximisingPlayer) {
        return this.rand.nextFloat();
    }
}
