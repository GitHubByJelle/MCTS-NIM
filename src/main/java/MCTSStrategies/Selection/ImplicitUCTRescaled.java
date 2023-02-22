//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import MCTSStrategies.Rescaler.Linear;
import MCTSStrategies.Rescaler.Rescaler;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values.
 * However, instead of directly using the estimated values by the GameStateEvaluator, the values first get
 * rescaled by a Rescaler
 */
public class ImplicitUCTRescaled extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /**
     * Rescaler used to rescale the estimated values by the GameStateEvaluator before using them in implicit UCT
     */
    Rescaler rescaler;

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (influence=0.4, exploration=sqrt(2), with a linear rescaler) (similar to
     * normal implicit UCT)
     */
    public ImplicitUCTRescaled() {
        this(0.4, Math.sqrt(2.0), new Linear());
    }


    /**
     * Constructor with influence of the implicit minimax value and exploration constant as input
     *
     * @param influenceEstimatedMinimax Influence of the implicit minimax value
     * @param explorationConstant       Exploration constant
     * @param rescaler                  Rescaler used to rescale the estimated values
     */
    public ImplicitUCTRescaled(double influenceEstimatedMinimax, double explorationConstant, Rescaler rescaler) {
        this.explorationConstant = explorationConstant;
        this.influenceEstimatedMinimax = influenceEstimatedMinimax;
        this.rescaler = rescaler;
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
        double bestValue = Double.NEGATIVE_INFINITY;
        int numBestFound = 0;
        double parentLog = Math.log((double) Math.max(1, current.sumLegalChildVisits()));
        int numChildren = current.numLegalMoves();
        State state = current.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = current.valueEstimateUnvisitedChildren(moverAgent);

        // Determine all estimated values
        double[] estimatedValues = new double[numChildren];
        for (int i = 0; i < numChildren; i++) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                estimatedValues[i] = ((implicitNode) current).getInitialEstimatedValue(i); // Own perspective
            } else {
                estimatedValues[i] = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }
        }

        // Rescales the estimated values
        estimatedValues = this.rescaler.rescale(estimatedValues);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        for (int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double) numVisits);
            }

            double uctValue = (1 - this.influenceEstimatedMinimax) * exploit +
                    this.influenceEstimatedMinimax * estimatedValues[i] +
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
