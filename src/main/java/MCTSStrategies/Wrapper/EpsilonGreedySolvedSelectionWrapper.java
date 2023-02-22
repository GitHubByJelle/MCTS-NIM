package MCTSStrategies.Wrapper;

import main.collections.FastArrayList;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;
import search.mcts.selection.SelectionStrategy;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Wrapper for selection strategy of MCTS that selects a random child with a probability. If a proven nodes are
 * found, a random proven node is selected, else just a random move is selected.
 */
public class EpsilonGreedySolvedSelectionWrapper implements SelectionStrategy {

    //-------------------------------------------------------------------------

    /**
     * The used selection strategy
     */
    public SelectionStrategy selectionStrategy;

    /**
     * Probability of playing a random move (value between 0 and 1)
     */
    float epsilon;

    //-------------------------------------------------------------------------

    /**
     * Constructor for the epsilon greedy move selection
     *
     * @param selectionStrategy The used selection strategy
     * @param epsilon           Probability of playing a random move (value between 0 and 1)
     */
    public EpsilonGreedySolvedSelectionWrapper(SelectionStrategy selectionStrategy, float epsilon) {
        this.selectionStrategy = selectionStrategy;
        this.epsilon = epsilon;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on epsilon-greedy preferring
     * proven nodes
     *
     * @param mcts    Ludii's MCTS class
     * @param current node representing the current game state
     * @return The index of next "best" move
     */
    @Override
    public int select(MCTS mcts, BaseNode current) {
        // If below the epsilon
        if (ThreadLocalRandom.current().nextFloat() < this.epsilon) {
            int numChildren = current.numLegalMoves();
            State state = current.contextRef().state();
            int moverAgent = state.playerToAgent(state.mover());
            FastArrayList<Integer> consideredIndices = new FastArrayList<>();
            boolean provenWin = false;

            // Take random child, but if a completed node is available take a random completed node
            for (int i = 0; i < numChildren; i++) {
                BaseNode child = current.childForNthLegalMove(i);
                if (!provenWin) {
                    if (child != null) {
                        // If it is a win, remove all found indices, and only use proven wins
                        // If it is a loss, ignore
                        if (child.totalScore(moverAgent) == Double.POSITIVE_INFINITY) {
                            provenWin = true;
                            consideredIndices.clear();
                        } else if (child.totalScore(moverAgent) == Double.NEGATIVE_INFINITY) {
                            continue;
                        }
                    }

                    consideredIndices.add(i);
                } else {
                    // If it is a win as well, add to the indices
                    if (child != null && child.totalScore(moverAgent) == Double.POSITIVE_INFINITY) {
                        consideredIndices.add(i);
                    }
                }
            }

            // If all children are proven losses, take a random one, and stop search.
            if (consideredIndices.size() == 0) {
                if (current.parent() == null) {
                    ((Agents.MCTS) mcts).setStop(true);
                }
                return consideredIndices.get(ThreadLocalRandom.current().nextInt(numChildren));
            }

            // Return random indices
            return consideredIndices.get(ThreadLocalRandom.current().nextInt(consideredIndices.size()));
        } else {
            // Perform normal selection
            return this.selectionStrategy.select(mcts, current);
        }
    }

    /**
     * Flags for data the selection strategy's Backpropagation wants to track.
     *
     * @return Additional flags for data the selection strategy's Backpropagation wants to track.
     */
    @Override
    public int backpropFlags() {
        return this.selectionStrategy.backpropFlags();
    }

    /**
     * Flags for data the selection strategy's expansion wants to track.
     *
     * @return Additional flags for data the selection strategy's expansion wants to track.
     */
    @Override
    public int expansionFlags() {
        return this.selectionStrategy.expansionFlags();
    }

    /**
     * Copies the customise method of the initialised selection strategy class
     *
     * @param strings inputs to customise
     */
    @Override
    public void customise(String[] strings) {
        this.selectionStrategy.customise(strings);
    }
}
