//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import other.move.Move;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.backpropagation.BackpropagationStrategy;
import search.mcts.nodes.BaseNode;
import search.mcts.selection.SelectionStrategy;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of UCT, Progressive History and GRAVE
 */
public class ProgressiveHistoryGRAVE implements SelectionStrategy {

    //-------------------------------------------------------------------------

    /** Weight of Progressive history (based on intial value Ludii implementation, see ProgressiveBias) */
    protected final double progressiveWeight;

    /** Threshold number of playouts that a node must have had for its AMAF values to be used */
    protected final int ref;

    /** Hyperparameter used in computation of weight for AMAF term */
    protected final double bias;

    /** Exploration constant */
    protected double explorationConstant;

    /** Reference node in current MCTS simulation (one per thread, in case of multi-threaded MCTS) */
    protected ThreadLocal<BaseNode> currentRefNode = ThreadLocal.withInitial(() -> null);

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (Threshold number=100, weight used in AMAF=10.e-6, progressive weights=0.3
     * and exploration=sqrt(2))
     */
    public ProgressiveHistoryGRAVE() {
        this(100, 10.e-6, 3.0, Math.sqrt(2.0));
    }

    /**
     * Constructor with threshold number, weight used in computation AMAF, weight of progressive history and
     * exploration constant as input
     *
     * @param ref Threshold number of playouts that a node must have had for its AMAF values to be used
     * @param bias Hyperparameter used in computation of weight for AMAF term
     * @param progressiveWeight Weight of Progressive history
     * @param explorationConstant Exploration constant
     */
    public ProgressiveHistoryGRAVE(final int ref, final double bias,
                                   final double progressiveWeight, final double explorationConstant) {
        this.ref = ref;
        this.bias = bias;
        this.progressiveWeight = progressiveWeight;
        this.explorationConstant = explorationConstant;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on Progressive
     * History and GRAVE
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

        // Set current reference node for current thread
        if (currentRefNode.get() == null || current.numVisits() > ref || current.parent() == null)
            currentRefNode.set(current);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double explore;
        double meanScore;
        double meanAMAF;
        double beta;
        int numVisits;
        double meanGlobalActionScore;
        for(int i = 0; i < numChildren; ++i) {
            BaseNode child = current.childForNthLegalMove(i);
            final Move move = current.nthLegalMove(i);
            final MCTS.ActionStatistics actionStats = mcts.getOrCreateActionStatsEntry(new MCTS.MoveKey(move, current.contextRef().trial().numMoves()));
            if (actionStats.visitCount == 0)
                meanGlobalActionScore = unvisitedValueEstimate;
            else
                meanGlobalActionScore = actionStats.accumulatedScore / actionStats.visitCount;

            if (child == null) {
                meanScore = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                numVisits = 0;
                meanAMAF = 0.0;
                beta = 0.0;
            } else {
                meanScore = child.exploitationScore(moverAgent);
                numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double)numVisits);

                final BaseNode.NodeStatistics graveStats = currentRefNode.get().graveStats(new MCTS.MoveKey(move, current.contextRef().trial().numMoves()));

                if (graveStats == null)
                {
                    // In single-threaded MCTS this should always be a bug,
                    // but in multi-threaded MCTS it can happen
                    meanAMAF = 0.0;
                    beta = 0.0;
                }
                else {
                    final double graveScore = graveStats.accumulatedScore;
                    final int graveVisits = graveStats.visitCount;
                    meanAMAF = graveScore / graveVisits;
                    beta = graveVisits / (graveVisits + numVisits + bias * graveVisits * numVisits);
                }
            }

            double graveValue = (1.0 - beta) * meanScore + beta * meanAMAF;
            final double uctPhValue = graveValue + explorationConstant * explore
                    + meanGlobalActionScore * (progressiveWeight / ((1.0 - meanScore) * numVisits + 1));

            if (uctPhValue > bestValue) {
                bestValue = uctPhValue;
                bestIdx = i;
                numBestFound = 1;
            } else if (uctPhValue == bestValue) {
                int randomInt = ThreadLocalRandom.current().nextInt();
                ++numBestFound;
                if (randomInt % numBestFound == 0) {
                    bestIdx = i;
                }
            }
        }


        // This can help garbage collector to clean up a bit more easily
        if (current.childForNthLegalMove(bestIdx) == null)
            currentRefNode.set(null);

        return bestIdx;
    }

    /**
     * @return Flags indicating stats that should be backpropagated
     */
    public int backpropFlags() {
        return BackpropagationStrategy.GLOBAL_ACTION_STATS | BackpropagationStrategy.GRAVE_STATS;
    }

    /**
     * @return Flags indicating special things we want to do when expanding nodes
     */
    public int expansionFlags() {
        return 0;
    }

    /**
     * Customize the selection strategy based on a list of given string inputs.
     * Please note, this isn't adapted for all parameters used.
     *
     * @param inputs indicating what to customise
     */
    public void customise(String[] inputs) {
        if (inputs.length > 1) {
            for(int i = 1; i < inputs.length; ++i) {
                String input = inputs[i];
                if (input.startsWith("explorationconstant=")) {
                    this.explorationConstant = Double.parseDouble(input.substring("explorationconstant=".length()));
                } else {
                    System.err.println("UCB1 ignores unknown customisation: " + input);
                }
            }
        }
    }
}
