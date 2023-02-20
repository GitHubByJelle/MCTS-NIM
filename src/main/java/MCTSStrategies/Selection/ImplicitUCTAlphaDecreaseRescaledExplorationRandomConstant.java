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
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values, with a
 * decreasing alpha. However, instead of multiplying all children with the same exploration constant, a softmax
 * with temperature C/n_c is used to transform the estimated values to probabilities which are multiplied with the
 * exploration. The uct is multiplied with 1+random constant to increase exploration.
 */
public class ImplicitUCTAlphaDecreaseRescaledExplorationRandomConstant extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /** Softmax rescaler */
    Softmax rescaler;

    /** Slope of the change of the influence of the estimated value */
    protected double slope;

    /** Random constant multiplier (uct is multiplied with 1+random constant) */
    protected double randomConstant;

    //-------------------------------------------------------------------------

    /**
     * Constructor with influence of the implicit minimax value, exploration constant, rescaler, slope and random
     * constant as input
     *
     * @param initialInfluenceEstimatedMinimax Initial influence of the implicit minimax value
     * @param explorationConstant Exploration constant
     * @param rescaler Softmax rescaler
     * @param slope Slope of increase of alpha
     * @param randomConstant Random constant multiplier (uct is multiplied with 1+random constant)
     */
    public ImplicitUCTAlphaDecreaseRescaledExplorationRandomConstant(double initialInfluenceEstimatedMinimax,
                                                                     double explorationConstant,
                                                                     Softmax rescaler,
                                                                     double slope, double randomConstant) {
        super(initialInfluenceEstimatedMinimax, explorationConstant);

        this.rescaler = rescaler;
        this.slope = slope;
        this.randomConstant = randomConstant;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on implicit UCT with an increasing alpha
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
        double parentVisits = (double)Math.max(1, current.sumLegalChildVisits());
        double parentLog = Math.log(parentVisits);
        int numChildren = current.numLegalMoves();
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
        double[] explorationProbs = this.rescaler.rescale(estimatedValues, this.explorationConstant / parentVisits);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        int numVisits;
        double alpha;
        for(int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = explorationProbs[i] * Math.sqrt(parentLog);
                numVisits = 0;
            } else {
                exploit = child.exploitationScore(moverAgent);
                numVisits = child.numVisits() + child.numVirtualVisits();
                explore = explorationProbs[i] * Math.sqrt(parentLog / (double)numVisits);
            }

            alpha = this.adjustAlpha(this.influenceEstimatedMinimax, numVisits);
            double uctValue = (1 - alpha) *  exploit +
                    alpha * estimatedValues[i] +
                    explore;

            uctValue *= (1 + ThreadLocalRandom.current().nextDouble(0, this.randomConstant));

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

    /**
     * Adjust alpha to increase over-time
     *
     * @param initialAlpha Initial influence of the estimated values
     * @param numVisits Number of visits to current node
     * @return Adjusted alpha
     */
    protected double adjustAlpha(double initialAlpha, int numVisits){
        return Math.max(0, initialAlpha - this.slope * numVisits * initialAlpha);
    }
}
