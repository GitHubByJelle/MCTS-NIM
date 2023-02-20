//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import main.collections.FVector;
import other.context.Context;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.backpropagation.BackpropagationStrategy;
import search.mcts.nodes.BaseNode;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values when the
 * visits are above a given threshold. If the visits are too small, moves are selected based on epsilon-greedy MAST.
 */
public class ImplicitUCTThreshold extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /** Threshold to switch from MAST to implicit UCT */
    protected int threshold;

    /** Epsilon used for epsilon-greedy MAST */
    protected float epsilon = .05f;

    //-------------------------------------------------------------------------

    /**
     * Constructor with influence of the implicit minimax value, exploration constant and threshold to switch as input
     *
     * @param influenceEstimatedMinimax Influence of the implicit minimax value
     * @param explorationConstant Exploration constant
     * @param threshold Threshold to switch from MAST to implicit UCT
     */
    public ImplicitUCTThreshold(double influenceEstimatedMinimax, double explorationConstant, int threshold) {
        this.explorationConstant = explorationConstant;
        this.influenceEstimatedMinimax = influenceEstimatedMinimax;
        this.threshold = threshold;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on epsilon-greedy MAST below the
     * given threshold and based on implicit UCT above the given threshold
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

        // Check if below threshold, if so select with epsilon-greedy MAST
        if (current.numVisits() < this.threshold){
            // If below epsilon return random
            if (ThreadLocalRandom.current().nextFloat() < this.epsilon){
                return ThreadLocalRandom.current().nextInt(numChildren);
            }
            // Else return based on MAST
            return MASTIndexSelector.selectIndex(current, mcts);
        }

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        double estimatedValue;
        for(int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                estimatedValue = ((implicitNode)current).getInitialEstimatedValue(i); // Own perspective
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double)numVisits);
                estimatedValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }

            double uctValue = (1 - this.influenceEstimatedMinimax) *  exploit +
                    this.influenceEstimatedMinimax * estimatedValue +
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
     * @return Flags indicating stats that should be backpropagated
     */
    public int backpropFlags() {
        return BackpropagationStrategy.GLOBAL_ACTION_STATS;
    }

    /**
     * Playout Index Selector for MAST (NOTE: this one is just greedy, need
     * to put an epsilon-greedy wrapper around it for epsilon-greedy behaviour).
     *
     * @author Dennis Soemers, adapted by Jelle Jansen
     */
    protected static class MASTIndexSelector
    {
        public static int selectIndex
                (
                        final BaseNode current,
                        final MCTS mcts
                )
        {
            Context context = current.contextRef();
            int numLegalMoves = current.numLegalMoves();
            final FVector actionScores = new FVector(numLegalMoves);
            for (int i = 0; i < numLegalMoves; ++i)
            {
                final MCTS.ActionStatistics actionStats = mcts.getOrCreateActionStatsEntry(
                        new MCTS.MoveKey(current.nthLegalMove(i), context.trial().numMoves()));

                if (actionStats.visitCount > 0.0)
                    actionScores.set(i, (float) (actionStats.accumulatedScore / actionStats.visitCount));
                else
                    actionScores.set(i, 1.f);
            }

            return actionScores.argMaxRand();
        }
    }
}
