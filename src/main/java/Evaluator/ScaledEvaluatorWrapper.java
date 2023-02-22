package Evaluator;

import other.context.Context;

/**
 * Generates a wrapper for the "GameStateEvaluator" class which can be used to scale the output of a
 * "GameStateEvaluator". It will divide the original output by a given value.
 */
public class ScaledEvaluatorWrapper extends GameStateEvaluator {

    //-------------------------------------------------------------------------

    /**
     * GameStateEvaluator that needs to be scaled
     */
    GameStateEvaluator gameStateEvaluator;

    /**
     * Value that gets being used to divide the original output (slope)
     */
    float divideValue = 1;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the GameStateEvaluator that needs to be scaled and the value that gets being used
     * to divide the original output as input.
     *
     * @param gameStateEvaluator GameStateEvaluator that needs to be scaled
     * @param divideValue        Value that gets being used to divide the original output as input (slope)
     */
    public ScaledEvaluatorWrapper(GameStateEvaluator gameStateEvaluator, float divideValue) {
        this.gameStateEvaluator = gameStateEvaluator;
        this.divideValue = divideValue;
    }

    /**
     * Uses the original GameStateEvaluator to evaluate the context, but divides the output with "divideValue"
     * The expected output should be between [-1, 1]
     *
     * @param context          Ludii's context
     * @param maximisingPlayer The maximising player
     * @return The divided output of the original GameStateEvaluator
     */
    @Override
    public float evaluate(Context context, int maximisingPlayer) {
        // Get original output and scale
        float value = this.gameStateEvaluator.evaluate(context, maximisingPlayer) / divideValue;

        // Check if value doesn't exceed [-1, 1]
        if (value > 1 || value < -1) {
            throw new RuntimeException("Bounds exceed 1 (" + value +
                    "). Score: " + (value * this.divideValue));
        }

        return value;
    }
}
