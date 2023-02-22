package Evaluator;

import other.RankUtils;
import other.context.Context;

/**
 * Generates a terminal state evaluator based on the additive depth heuristic proposed by Cohen-Solal in:
 * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without knowledge.
 * arXiv preprint arXiv:2008.01188.
 * <p>
 * As opposed to the work of Cohen-Solal, this implementation returns a value between [-1 and 1], such that no changes
 * need to be made to the architecture of the NN
 */
public class AdditiveDepthTerminalStateEvaluator extends GameStateEvaluator {

    //-------------------------------------------------------------------------

    /**
     * Maximum number of plies for the total game
     */
    private int maxPly;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the maximum number of plies for the total game as input.
     *
     * @param maxPly Maximum number of plies for the total game
     */
    public AdditiveDepthTerminalStateEvaluator(int maxPly) {
        this.maxPly = maxPly;
    }

    /**
     * Evaluates the current terminal state using the additive depth heuristic as proposed by Cohen-Solal.
     * It will return a 0 if the state is non-terminal
     *
     * @param context          Ludii's context
     * @param maximisingPlayer The maximising player
     * @return A float value indicating how good the game state is (higher is better)
     */
    @Override
    public float evaluate(Context context, int maximisingPlayer) {
        // Determine the win
        double[] ranks = RankUtils.utilities(context);

        // Determine l
        float l = this.maxPly - context.trial().moveNumber();

        // Check if l is larger than one
        l = l < 1 ? 1 / this.maxPly : l / this.maxPly;

        // Return the l for a win and -l for a loss
        return (float) ranks[context.state().playerToAgent(maximisingPlayer)] * l;
    }
}
