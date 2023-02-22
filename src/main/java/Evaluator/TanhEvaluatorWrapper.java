package Evaluator;

import other.context.Context;

/**
 * Generates a wrapper for the "GameStateEvaluator" class which can be used to scale the output of a
 * "GameStateEvaluator" with tanh
 */
public class TanhEvaluatorWrapper extends GameStateEvaluator {

    //-------------------------------------------------------------------------

    /**
     * GameStateEvaluator that needs to be scaled
     */
    GameStateEvaluator gameStateEvaluator;

    /**
     * Value that gets being used to divide the original output (slope)
     */
    float divideValue = 1;

    /**
     * Maximum bound of original value
     */
    float maxBound;

    /**
     * Minimum bound of original value
     */
    float minBound;

    //-------------------------------------------------------------------------

    /**
     * Constructor requiring the original GameStateEvaluator as input, division value is set to 1 and
     * bounds to [-inf, inf]
     *
     * @param gameStateEvaluator GameStateEvaluator that needs to be scaled
     */
    public TanhEvaluatorWrapper(GameStateEvaluator gameStateEvaluator) {
        this(gameStateEvaluator, 1, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
    }

    /**
     * Constructor requiring the original GameStateEvaluator, the division value and the bounds
     *
     * @param gameStateEvaluator GameStateEvaluator that needs to be scaled
     * @param divideValue        Value that gets being used to divide the original output (slope)
     * @param maxBound           Maximum bound of original value
     * @param minBound           Minimum bound of original value
     */
    public TanhEvaluatorWrapper(GameStateEvaluator gameStateEvaluator, float divideValue, float maxBound, float minBound) {
        this.gameStateEvaluator = gameStateEvaluator;
        this.divideValue = divideValue;
        this.maxBound = maxBound;
        this.minBound = minBound;
    }

    /**
     * Uses the original GameStateEvaluator to evaluate the context, but scales the output
     * between [-1, 1] using a tanh and a slope
     *
     * @param context          Ludii's context
     * @param maximisingPlayer The maximising player
     * @return Scaled output of the original GameStateEvaluator
     */
    @Override
    public float evaluate(Context context, int maximisingPlayer) {
        // Get original output
        float score = this.gameStateEvaluator.evaluate(context, maximisingPlayer);

        // Check if score doesn't exceed bounds, if so set equal to bound
        if (score > this.maxBound) {
            score = this.maxBound;
        } else if (score < this.minBound) {
            score = this.minBound;
        }

        // Divide value (slope) and get tanh
        return (float) Math.tanh(score / divideValue);
    }
}
