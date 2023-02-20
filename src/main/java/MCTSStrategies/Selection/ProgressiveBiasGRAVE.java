//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import other.move.Move;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.backpropagation.BackpropagationStrategy;
import search.mcts.nodes.BaseNode;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of Progressive Bias using the GameStateEvaluator
 * and GRAVE. Instead of using the direct evaluations, it uses implicit minimax backpropagation to evaluate the states (as done with
 * Implicit UCT).
 */
public class ProgressiveBiasGRAVE extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /** Weight of Progressive bias (based on intial value Ludii implementation, see ProgressiveBias) */
    protected final double progressiveWeight = 10;

    /** Threshold number of playouts that a node must have had for its AMAF values to be used */
    protected final int ref;

    /** Hyperparameter used in computation of weight for AMAF term */
    protected final double bias;

    /** Reference node in current MCTS simulation (one per thread, in case of multi-threaded MCTS) */
    protected ThreadLocal<BaseNode> currentRefNode = ThreadLocal.withInitial(() -> null);

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (Threshold number=100, weight used in AMAF=10.e-6 and exploration=sqrt(2))
     */
    public ProgressiveBiasGRAVE() {
        this(100, 10.e-6, Math.sqrt(2.0));
    }

    /**
     * Constructor with threshold number, weight used in computation AMAF, and exploration constant as input
     *
     * @param ref Threshold number of playouts that a node must have had for its AMAF values to be used
     * @param bias Hyperparameter used in computation of weight for AMAF term
     * @param explorationConstant Exploration constant
     */
    public ProgressiveBiasGRAVE(final int ref, final double bias, double explorationConstant) {
        this.ref = ref;
        this.bias = bias;
        this.explorationConstant = explorationConstant;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on a combination of Progressive Bias
     * and GRAVE as implemented by Ludii. However, this implementation uses the GameStateEvaluators, which makes it able to run Progressive Bias
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

        // Set current reference node for current thread
        if (currentRefNode.get() == null || current.numVisits() > ref || current.parent() == null)
            currentRefNode.set(current);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double explore;
        double estimatedValue;
        double meanScore;
        double meanAMAF;
        double beta;
        double estimatedScore;
        for(int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                meanScore = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                meanAMAF = 0.0;
                beta = 0.0;
                estimatedScore = this.progressiveWeight *
                        ((implicitNode)current).getInitialEstimatedValue(i); // Own perspective
            } else {
                meanScore = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double)numVisits);
                estimatedValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
                estimatedScore = (this.progressiveWeight * estimatedValue) / numVisits;

                final Move move = child.parentMove();
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
            final double uctGraveValue = graveValue + explorationConstant * explore + estimatedScore;

            if (uctGraveValue > bestValue) {
                bestValue = uctGraveValue;
                bestIdx = i;
                numBestFound = 1;
            } else if (uctGraveValue == bestValue) {
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

    public int backpropFlags() {
        return BackpropagationStrategy.GRAVE_STATS;
    }
}
