//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import MCTSStrategies.Rescaler.Softmax;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values.
 * However, instead the exploration part is replaced with probabilities calculated by a softmax with temperature
 * 1/n_c based on the estimated values.
 */
public class ImplicitUCTRescaledExploration4 extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /** Softmax rescaler */
    Softmax rescaler;

    //-------------------------------------------------------------------------

    /**
     * Constructor with influence of the implicit minimax value and softmax rescaler as input
     *
     * @param influenceEstimatedMinimax Influence of the implicit minimax value
     * @param rescaler Rescaler used to rescale the estimated values
     */
    public ImplicitUCTRescaledExploration4(double influenceEstimatedMinimax, Softmax rescaler) {
        this.influenceEstimatedMinimax = influenceEstimatedMinimax;
        this.rescaler = rescaler;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on implicit UCT with
     * modified exploration. The exploration is replaced with probabilities determined by a
     * softmax with temperature 1/n_c on the estimated values
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
        int numChildren = current.numLegalMoves();
        double parentVisits = (double)Math.max(1, current.sumLegalChildVisits());
        State state = current.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = current.valueEstimateUnvisitedChildren(moverAgent);

        // Determine all estimated values
        double[] estimatedValues = new double[numChildren];
        for (int i = 0; i < numChildren; i++) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                estimatedValues[i] = ((implicitNode)current).getInitialEstimatedValue(i); // Own perspective
            } else {
                estimatedValues[i] = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }
        }

        // Determine exploration probabilities based on the softmax
        double[] explorationProbs = this.rescaler.rescale(estimatedValues, 1 / parentVisits);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        for(int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = explorationProbs[i];
            } else {
                exploit = child.exploitationScore(moverAgent);
                explore = explorationProbs[i];
            }

            double uctValue = (1 - this.influenceEstimatedMinimax) *  exploit +
                    this.influenceEstimatedMinimax * estimatedValues[i] +
                    explore;
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
