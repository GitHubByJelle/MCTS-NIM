package Evaluator;

import other.RankUtils;
import other.context.Context;

/**
 * Generates a terminal state evaluator that returns 9999999 for a win, 0 for a draw and -9999999 for a loss
 */
public class MaxClassicTerminalStateEvaluator extends GameStateEvaluator {

    /**
     * Constructor without parameters
     */
    public MaxClassicTerminalStateEvaluator() {
    }

    /**
     * Evaluates the current terminal state with a 9999999, 0 or -9999999 for a win, draw or loss for the
     * maximising player, respectively. If the game state isn't terminal, a 0 is returned.
     *
     * @param context          Ludii's context
     * @param maximisingPlayer The maximising player
     * @return A float value indicating how good the game state is (higher is better)
     */
    @Override
    public float evaluate(Context context, int maximisingPlayer) {
        // Evaluation of terminal states, e.g. gain game (first player point of view)
        double[] ranks = RankUtils.utilities(context);
        return (float) ranks[context.state().playerToAgent(maximisingPlayer)] * 9999999;
    }
}
