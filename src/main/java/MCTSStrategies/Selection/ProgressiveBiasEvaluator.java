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
 * Selection strategy which selects the child based on Progressive Bias using the GameStateEvaluator. Instead
 * of using the direct evaluations, it uses implicit minimax backpropagation to evaluate the states (as done with
 * Implicit UCT).
 */
public class ProgressiveBiasEvaluator extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /** Weight of Progressive bias (based on intial value Ludii implementation, see ProgressiveBias) */
    protected final double progressiveWeight = 10;

    //-------------------------------------------------------------------------


    /**
     * Constructor with no inputs (exploration=sqrt(2))
     */
    public ProgressiveBiasEvaluator() {
        this(Math.sqrt(2.0));
    }

    /**
     * Constructor with exploration constant as input
     *
     * @param explorationConstant Exploration constant
     */
    public ProgressiveBiasEvaluator(double explorationConstant) {
        this.explorationConstant = explorationConstant;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on Progressive Bias as implemented by
     * Ludii. However, this implementation uses the GameStateEvaluators, which makes it able to run Progressive Bias
     * with multiple evaluators. On top of that, it uses the minimax backpropagated values as implicit UCT.
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


            final double uctValue = exploit + explorationConstant * explore + estimatedScore;

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
