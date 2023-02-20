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
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values.
 * However, instead of using the estimated value, the different between the mean of the estimated value of all
 * children and the estimated value is used
 */
public class ImplicitUCTDifferenceV extends ImplicitUCT {

    /**
     * Constructor with influence of the implicit minimax value and exploration constant as input
     *
     * @param influenceEstimatedMinimax Influence of the implicit minimax value
     * @param explorationConstant Exploration constant
     */
    public ImplicitUCTDifferenceV(double influenceEstimatedMinimax, double explorationConstant) {
        this.explorationConstant = explorationConstant;
        this.influenceEstimatedMinimax = influenceEstimatedMinimax;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on implicit UCT while using
     * the different to the mean instead of the actual value
     *
     * @param mcts Ludii's MCTS class
     * @param current node representing the current game state
     * @return The index of next "best" move
     */
    public int select(MCTS mcts, BaseNode current) {
        // Initialise needed variables
        int bestIdx = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        int numBestFound = 0;
        double parentLog = Math.log((double)Math.max(1, current.sumLegalChildVisits()));
        int numChildren = current.numLegalMoves();
        State state = current.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = current.valueEstimateUnvisitedChildren(moverAgent);
        double estimatedSum = 0;

        // Extract all estimated values
        double[] estimatedValues = new double[numChildren];
        for (int i = 0; i < numChildren; i++) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                estimatedValues[i] = ((implicitNode)current).getInitialEstimatedValue(i); // Own perspective
            } else {
                estimatedValues[i] = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }

            estimatedSum += estimatedValues[i];
        }

        // Calculate the mean of the estimated value of all children
        double estimatedAverage = estimatedSum / numChildren;

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        for(int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double)numVisits);
            }

            double uctValue = (1 - this.influenceEstimatedMinimax) *  exploit +
                    this.influenceEstimatedMinimax * (estimatedValues[i] - estimatedAverage) +
                    this.explorationConstant * explore;

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
