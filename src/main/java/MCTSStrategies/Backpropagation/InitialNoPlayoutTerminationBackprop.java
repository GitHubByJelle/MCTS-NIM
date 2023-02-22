//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Backpropagation;

import MCTSStrategies.Node.implicitNode;
import other.context.Context;
import other.move.Move;
import search.mcts.MCTS;
import search.mcts.backpropagation.BackpropagationStrategy;
import search.mcts.nodes.BaseNode;

// Backpropagates the value of the evaluation function after a fixed number of steps
// PLEASE NOTE: ASSUMPTION IS MADE OF TWO PLAYERS (can save a lot of time when using NN as leaf evaluator)

/**
 * Backpropagates the best estimated value of the context of the implicit node. Can only be used in combination with implicitUCT.
 * PLEASE NOTE: ASSUMPTION IS MADE OF TWO PLAYERS (can save a lot of time when using NN as leaf evaluator)
 */
public class InitialNoPlayoutTerminationBackprop extends BackpropagationStrategy {
    /**
     * Constructor requiring no inputs
     */
    public InitialNoPlayoutTerminationBackprop() {
    }

    /**
     * Extracts the array of utilities that we want to backpropagate.
     * Instead of returning a win or loss it returns the estimated value of the context of the last encountered node
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
            // Initialise all needed objects
            double value = 0;
            Move parentMove = startNode.parentMove();
            BaseNode parentNode = startNode.parent();

            // Determine multiplier based on player to move (for parent node)
            double multiplier = parentNode.contextRef().state().playerToAgent(parentNode.contextRef().state().mover()) == 1 ?
                    1 : -1;

            // For all children (of parent, so all siblings of startNode), extract the
            // initial estimated value of the node
            int numChildren = parentNode.numLegalMoves();
            for (int i = 0; i < numChildren; i++) {
                if (parentNode.childForNthLegalMove(i) == startNode) {
                    value = ((implicitNode) parentNode).getInitialEstimatedValue(i) * multiplier;
                    break;
                }
            }

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
}
