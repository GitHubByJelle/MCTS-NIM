//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the implicit minimax values (from implicit MCTS nodes) epsilon greedy
 */
public class EpsilonGreedy extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /** Probability of playing a random move (value between 0 and 1) */
    float epsilon;

    //-------------------------------------------------------------------------

    /**
     * Constructor with no input (epsilon = 0.05 is used)
     */
    public EpsilonGreedy() {
        this(0.05f);
    }

    /**
     * Constructor with epsilon as input
     *
     * @param epsilon Probability of playing a random move (value between 0 and 1)
     */
    public EpsilonGreedy(float epsilon) {
        super(1, 0);

        this.epsilon = epsilon;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on epsilon-greedy
     *
     * @param mcts Ludii's MCTS class
     * @param current node representing the current game state
     * @return The index of next "best" move
     */
    public int select(MCTS mcts, BaseNode current) {
        // Initialise needed variables
        int bestIdx = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        int numChildren = current.numLegalMoves();
        State state = current.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());

        // If random number is lower than epsilon, select random move
        if (ThreadLocalRandom.current().nextFloat() < this.epsilon){
            return ThreadLocalRandom.current().nextInt(numChildren);
        }

        // Else select move with highest implicit minimax value
        double heuristicValue;
        for(int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                heuristicValue = ((implicitNode)current).getInitialEstimatedValue(i); // Own perspective
            } else {
                heuristicValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }

            if (heuristicValue > bestValue) {
                bestValue = heuristicValue;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /**
     * @return Flags indicating stats that should be backpropagated
     */
    public int backpropFlags() {
        return 0;
    }

    /**
     * @return Flags indicating special things we want to do when expanding nodes
     */
    public int expansionFlags() {
        return 0;
    }

    /**
     * Customize the selection strategy based on a list of given string inputs
     *
     * @param inputs indicating what to customise
     */
    public void customise(String[] inputs) {
        if (inputs.length > 1) {
            for(int i = 1; i < inputs.length; ++i) {
                String input = inputs[i];
                if (input.startsWith("explorationconstant=")) {
                    this.explorationConstant = Double.parseDouble(input.substring("explorationconstant=".length()));
                } else {
                    System.err.println("UCB1 ignores unknown customisation: " + input);
                }
            }
        }
    }
}
