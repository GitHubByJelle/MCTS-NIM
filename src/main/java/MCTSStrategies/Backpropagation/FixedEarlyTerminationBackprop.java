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

// Backpropagates the value of the evaluation function after a fixed number of steps
// PLEASE NOTE: ASSUMPTION IS MADE OF TWO PLAYERS (can save a lot of time when using NN as leaf evaluator)

/**
 * Backpropagates the value of the evaluation function after a fixed number of steps
 * PLEASE NOTE: ASSUMPTION IS MADE OF TWO PLAYERS (can save a lot of time when using NN as leaf evaluator)
 */
public class FixedEarlyTerminationBackprop extends BackpropagationStrategy {

    //-------------------------------------------------------------------------

    /**
     * GameStateEvaluator to evaluate leaf nodes
     */
    GameStateEvaluator leafEvaluator;

    //-------------------------------------------------------------------------

    /**
     * Constructor requiring no inputs
     */
    public FixedEarlyTerminationBackprop() {
    }

    /**
     * Computes the array of utilities that we want to backpropagate.
     * Instead of returning a win or loss it returns the value of an evaluation function, unless a terminal
     * node is found
     *
     * @param mcts            Ludii's mcts base class
     * @param startNode       The last seen node during the play-out
     * @param context         Ludii's context
     * @param utilities       Initial utilities value indicating null, the terminal value for player 1, and the terminal
     *                        value for player 2
     * @param numPlayoutMoves Number of moves made in play-out
     */
    public void computeUtilities(MCTS mcts, BaseNode startNode, Context context, double[] utilities, int numPlayoutMoves) {
        if (context.active()) {
            double value = this.leafEvaluator.evaluate(context, 1);
            utilities[1] = value;
            utilities[2] = -value;
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
     * Setter for the leaf evaluator used as backpropagated value
     *
     * @param leafEvaluator GameStateEvaluator for the leaf nodes
     */
    public void setLeafEvaluator(GameStateEvaluator leafEvaluator) {
        this.leafEvaluator = leafEvaluator;
    }
}
