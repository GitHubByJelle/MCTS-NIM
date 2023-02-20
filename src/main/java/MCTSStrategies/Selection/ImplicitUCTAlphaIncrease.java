//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;
import search.mcts.selection.SelectionStrategy;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values, with an
 * increasing alpha
 */
public class ImplicitUCTAlphaIncrease extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /** Slope of the change of the influence of the estimated value */
    protected double slope;

    //-------------------------------------------------------------------------

    /**
     * Constructor with influence of the implicit minimax value, exploration constant and slope as input
     *
     * @param initialInfluenceEstimatedMinimax Initial influence of the implicit minimax value
     * @param explorationConstant Exploration constant
     * @param slope Slope of increase of alpha
     */
    public ImplicitUCTAlphaIncrease(double initialInfluenceEstimatedMinimax, double explorationConstant, double slope) {
        super(initialInfluenceEstimatedMinimax, explorationConstant);

        this.slope = slope;
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
        double parentLog = Math.log((double)Math.max(1, current.sumLegalChildVisits()));
        int numChildren = current.numLegalMoves();
        State state = current.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = current.valueEstimateUnvisitedChildren(moverAgent);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        double estimatedValue;
        int numVisits;
        double alpha;
        for(int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                estimatedValue = ((implicitNode)current).getInitialEstimatedValue(i); // Own perspective
                numVisits = 0;
            } else {
                exploit = child.exploitationScore(moverAgent);
                numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double)numVisits);
                estimatedValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }

            alpha = this.adjustAlpha(this.influenceEstimatedMinimax, numVisits);
            double uctValue = (1 - alpha) *  exploit +
                    alpha * estimatedValue +
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

    /**
     * Adjust alpha to increase over-time
     *
     * @param initialAlpha Initial influence of the estimated values
     * @param numVisits Number of visits to current node
     * @return Adjusted alpha
     */
    protected double adjustAlpha(double initialAlpha, int numVisits){
        return Math.min(1, initialAlpha + this.slope * numVisits * (1-initialAlpha));
    }
}
