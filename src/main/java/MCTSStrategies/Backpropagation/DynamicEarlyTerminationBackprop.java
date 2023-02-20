//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Backpropagation;

import Evaluator.GameStateEvaluator;
import other.context.Context;
import search.mcts.MCTS;
import search.mcts.backpropagation.BackpropagationStrategy;
import search.mcts.nodes.BaseNode;

import java.util.Arrays;

// Backpropagates the value of the evaluation function after a fixed number of steps
// PLEASE NOTE: ASSUMPTION IS MADE OF TWO PLAYERS (can save a lot of time when using NN as leaf evaluator)

/**
 * When the play-out is continued until the evaluation of the game state encountered during the
 * play-out exceeds a predefined threshold, this backpropagation method can be used. When it passes the positive
 * threshold a win is returned for the first player (1), but when it passes the negative threshold a win
 * is returned for the second player.
 * PLEASE NOTE: ASSUMPTION IS MADE OF TWO PLAYERS (can save a lot of time when using NN as leaf evaluator)
 */
public class DynamicEarlyTerminationBackprop extends BackpropagationStrategy {

    //-------------------------------------------------------------------------

    /** GameStateEvaluator to evaluate leaf nodes */
    GameStateEvaluator leafEvaluator;

    /** Threshold of the play-out (before returning a win) */
    protected float threshold;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the threshold value as input
     *
     * @param threshold Threshold of the play-out (before returning a win)
     */
    public DynamicEarlyTerminationBackprop(float threshold) {
        this.threshold = threshold;
    }

    /**
     * Constructor with the threshold value as input
     *
     * @param threshold Threshold of the play-out (before returning a win)
     * @param leafEvaluator GameStateEvaluator for the leaf nodes
     */
    public DynamicEarlyTerminationBackprop(float threshold, GameStateEvaluator leafEvaluator) {
        this.threshold = threshold;

        this.leafEvaluator = leafEvaluator;
    }

    /**
     * Computes the array of utilities that we want to backpropagate.
     * When it passes the positive threshold a win is returned for the first
     * player (1), but when it passes the negative threshold a win is returned for the second player.
     * It is assumed that the game position passes the threshold positively or negatively.
     *
     * @param mcts Ludii's mcts base class
     * @param startNode The last seen node during the play-out
     * @param context Ludii's context
     * @param utilities Initial utilities value indicating null, the terminal value for player 1, and the terminal
     *                  value for player 2
     * @param numPlayoutMoves Number of moves made in play-out
     */
    public void computeUtilities(MCTS mcts, BaseNode startNode, Context context, double[] utilities, int numPlayoutMoves) {
        // If the game position isn't terminal
        if (context.active()) {
            // Determine estimated evaluation
            double value = this.leafEvaluator.evaluate(context, 1);

            // If it passes positively return win for player 1, and loss for player 2
            if (value > this.threshold){
                utilities[1] = 1;
                utilities[2] = -1;
            }
            // If it passes negatively return win for player 2, and loss for player 1
            else {
                utilities[1] = -1;
                utilities[2] = 1;
            }
        }
    }

    /**
     * Flags for data this Backpropagation wants to track.
     *
     * @return Additional flags for data this Backpropagation wants to track.
     */
    public int backpropagationFlags() {
        return 0;
    }

    /**
     * Setter for the leaf evaluator used in the dynamic evaluation when the leaf evaluator is not already set
     *
     * @param leafEvaluator GameStateEvaluator for the leaf nodes
     */
    public void setLeafEvaluator(GameStateEvaluator leafEvaluator) {
        if (this.leafEvaluator == null){
            this.leafEvaluator = leafEvaluator;
        }
    }
}
