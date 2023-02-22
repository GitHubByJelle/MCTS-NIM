//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;
import utils.Value;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values. However,
 * instead of selecting the best child, the top K children are compared with a second exploration value, to see
 * and selects the best child from that.
 */
public class ImplicitUCTTopKExploration extends ImplicitUCTTopKRandom {

    //-------------------------------------------------------------------------

    /**
     * Second exploration constant
     */
    protected double explorationConstantTwo;

    //-------------------------------------------------------------------------

    /**
     * Constructor with influence of the implicit minimax value, exploration constants and K as input
     *
     * @param influenceEstimatedMinimax Influence of the implicit minimax value
     * @param explorationConstantOne    First exploration constant
     * @param explorationConstantTwo    Second exploration constant
     * @param K                         Number of best-children to select for second comparison
     */
    public ImplicitUCTTopKExploration(double influenceEstimatedMinimax, double explorationConstantOne,
                                      double explorationConstantTwo, int K) {
        super(influenceEstimatedMinimax, explorationConstantOne, K);

        this.explorationConstantTwo = explorationConstantTwo;
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

        // Get the indices of the top K best UCT values
        int[] topKIndices = getTopKIndex(uctValues, this.K);

        // Determine the best uct value with the second exploration value
        int bestIdx = -1;
        double bestValue = -Value.INF;
        int numBestFound = 0;
        for (int i = 1; i < this.K; i++) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(topKIndices[i]);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                estimatedValue = ((implicitNode) current).getInitialEstimatedValue(topKIndices[i]); // Own perspective
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double) numVisits);
                estimatedValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }

            double uctValue = (1 - this.influenceEstimatedMinimax) * exploit +
                    this.influenceEstimatedMinimax * estimatedValue +
                    this.explorationConstantTwo * explore;

            if (uctValue > bestValue) {
                bestValue = uctValue;
                bestIdx = i;
                numBestFound = 1;
            } else if (uctValue == bestValue) {
                int randomInt = ThreadLocalRandom.current().nextInt();
                ++numBestFound;
                if (randomInt % numBestFound == 0) {
                    bestIdx = i;
                }
            }
        }

        return bestIdx;
    }
}
