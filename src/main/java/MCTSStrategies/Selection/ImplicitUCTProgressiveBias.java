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
 * Selection strategy which selects the child based on a combination of UCT, minimax backpropagated values and
 * Progressive Bias.
 */
public class ImplicitUCTProgressiveBias extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /** Weight of Progressive bias (based on intial value Ludii implementation, see ProgressiveBias) */
    protected final double progressiveWeight = 10;

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (influence=0.4 and exploration=sqrt(2))
     */
    public ImplicitUCTProgressiveBias() {
        this(0.8f, Math.sqrt(2.0));
    }

    /**
     * Constructor with influence of the implicit minimax value and exploration constant as input
     *
     * @param influenceEstimatedMinimax Influence of the implicit minimax value
     * @param explorationConstant Exploration constant
     */
    public ImplicitUCTProgressiveBias(final float influenceEstimatedMinimax, double explorationConstant) {
        this.influenceEstimatedMinimax = influenceEstimatedMinimax;
        this.explorationConstant = explorationConstant;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on implicit UCT and Progressive Bias
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
        double estimatedScore;
        for(int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);

            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                estimatedValue = ((implicitNode)current).getInitialEstimatedValue(i);
                estimatedScore = this.progressiveWeight * estimatedValue; // Own perspective
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double)numVisits);
                estimatedValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
                estimatedScore = (this.progressiveWeight * estimatedValue) / numVisits;
            }


            final double uctValue = (1 - this.influenceEstimatedMinimax) * exploit +
                    this.influenceEstimatedMinimax * estimatedValue + explorationConstant * explore + estimatedScore;

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
