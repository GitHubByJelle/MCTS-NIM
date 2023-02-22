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
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values. However,
 * instead of selecting the best child, a child is uniformly samples from the top K children.
 */
public class ImplicitUCTTopKRandom extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /**
     * Number of best-children to select
     */
    protected int K;

    //-------------------------------------------------------------------------

    /**
     * Constructor with influence of the implicit minimax value, exploration constant and K as input
     *
     * @param influenceEstimatedMinimax Influence of the implicit minimax value
     * @param explorationConstant       Exploration constant
     * @param K                         Number of best-children to select for the uniform sample
     */
    public ImplicitUCTTopKRandom(double influenceEstimatedMinimax, double explorationConstant, int K) {
        super(influenceEstimatedMinimax, explorationConstant);

        this.K = K;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on implicit UCT
     *
     * @param mcts    Ludii's MCTS class
     * @param current node representing the current game state
     * @return The index of next "best" move
     */
    public int select(MCTS mcts, BaseNode current) {
        // Initialise needed variables
        int bestIdx = -1;
        double parentLog = Math.log((double) Math.max(1, current.sumLegalChildVisits()));
        int numChildren = current.numLegalMoves();
        State state = current.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = current.valueEstimateUnvisitedChildren(moverAgent);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        double estimatedValue;
        double[] uctValues = new double[numChildren];
        for (int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                estimatedValue = ((implicitNode) current).getInitialEstimatedValue(i); // Own perspective
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double) numVisits);
                estimatedValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }

            uctValues[i] = (1 - this.influenceEstimatedMinimax) * exploit +
                    this.influenceEstimatedMinimax * estimatedValue +
                    this.explorationConstant * explore;
        }

        // Select random top element to play. Saves time when sorting since complexity is O(k*n)
        int k = ThreadLocalRandom.current().nextInt(1, this.K + 1);

        // Get the index of the kth-best UCT value
        return getTopKIndex(uctValues, k)[k - 1];
    }

    /**
     * Returns the indices of the top K maximum values of an array by using the
     * Partial Selection Sort algorithm. Complexity: O(n*k)
     *
     * @param array array to select the indices from
     * @param k     Number of top maximum values to get index from
     * @return Indices of top k maximum values of the given array
     */
    protected int[] getTopKIndex(double[] array, int k) {
        // Create requires objects
        int n = array.length;
        int[] topKIndices = new int[k];

        // Copy the values from the array
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            values[i] = array[i];
        }

        // For all k
        int maxIndex;
        double maxValue;
        for (int i = 0; i < k; i++) {
            maxIndex = i;
            maxValue = values[i];
            // Loop over (unsorted) values and select the best
            for (int j = i + 1; j < n; j++) {
                if (values[j] > maxValue) {
                    maxIndex = j;
                    maxValue = values[j];
                }
            }

            // Swap the values
            values[maxIndex] = values[i];
            values[i] = maxValue;

            // Save index
            topKIndices[i] = maxIndex;
        }

        return topKIndices;
    }
}
