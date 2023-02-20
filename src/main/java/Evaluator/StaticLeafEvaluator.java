package Evaluator;

import other.context.Context;

import java.util.Random;

/** Generates a leaf evaluator that always returns 0 */
public class StaticLeafEvaluator extends GameStateEvaluator {
    /**
     * Constructor requiring no inputs
     */
    public StaticLeafEvaluator() {}

    /**
     * Always returns 0
     *
     * @param context Ludii's context
     * @param maximisingPlayer The maximising player
     * @return 0
     */
    public float evaluate(Context context, int maximisingPlayer) {
        return 0;
    }
}
