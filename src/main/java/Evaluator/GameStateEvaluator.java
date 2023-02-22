package Evaluator;

import other.context.Context;

/**
 * Abstract class for all game state evaluators used
 * Allows to switch between Ludii's heuristic evaluators and DeepLearning4J evaluators easily
 */
public abstract class GameStateEvaluator {
    /**
     * Should be implemented in all evaluators, to evaluate game states
     *
     * @param context          Ludii's context
     * @param maximisingPlayer The maximising player
     * @return A float value indicating how good the game state is (higher is better).
     */
    public abstract float evaluate(Context context, int maximisingPlayer);
}

