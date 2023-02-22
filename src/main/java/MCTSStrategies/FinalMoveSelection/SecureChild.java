package MCTSStrategies.FinalMoveSelection;

import other.move.Move;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.finalmoveselection.FinalMoveSelectionStrategy;
import search.mcts.nodes.BaseNode;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selects move corresponding to the most secure child,
 * with an additional tie-breaker based randomness.
 * It combines the number of visits with average win rate found, as proposed in:
 * Winands, M. H., Björnsson, Y., & Saito, J. T. (2008, September). Monte-Carlo tree search solver. In
 * International Conference on Computers and Games (pp. 25-36). Springer, Berlin, Heidelberg.
 * <p>
 * Based on Ludii implementation
 */
public class SecureChild implements FinalMoveSelectionStrategy {

    //-------------------------------------------------------------------------

    private final double A;

    //-------------------------------------------------------------------------

    /**
     * Constructor without any inputs, A value of 1 is used.
     */
    public SecureChild() {
        this(1);
    }

    /**
     * Constructor without A as input.
     */
    public SecureChild(int A) {
        this.A = A;
    }

    /**
     * Selects the most secure child for the root node of the "actual" current game position
     *
     * @param mcts     Ludii's mcts class
     * @param rootNode Node of the "actual" current game position
     * @return Move that is most "secure"
     */
    public Move selectMove(MCTS mcts, BaseNode rootNode) {
        // Initialise all needed variables
        int bestIdx = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        int numBestFound = 0;
        int numChildren = rootNode.numLegalMoves();
        State state = rootNode.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = Double.NEGATIVE_INFINITY;

        // Look for all children which child is most secure with a random tie-breakers
        for (int i = 0; i < numChildren; ++i) {
            BaseNode child = rootNode.childForNthLegalMove(i);
            double value;
            if (child == null) {
                value = unvisitedValueEstimate;
            } else {
                value = child.totalScore(moverAgent) / child.numVisits() + 1 / Math.sqrt(child.numVisits());
            }

            // Random tie breaker
            if (value > bestValue) {
                bestValue = value;
                bestIdx = i;
                numBestFound = 1;
            } else if (value == bestValue) {
                int var10000 = ThreadLocalRandom.current().nextInt();
                ++numBestFound;
                if (var10000 % numBestFound == 0) {
                    bestIdx = i;
                }
            }
        }

        return rootNode.childForNthLegalMove(bestIdx).parentMove();
    }

    public void customise(final String[] inputs) {
        // Do nothing
    }
}
