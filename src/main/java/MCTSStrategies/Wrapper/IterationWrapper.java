package MCTSStrategies.Wrapper;

import other.move.Move;
import search.mcts.MCTS;
import search.mcts.finalmoveselection.FinalMoveSelectionStrategy;
import search.mcts.nodes.BaseNode;

/**
 * Wrapper class for FinalMoveSelectionStrategy which prints the number of iterations made during the search
 */
public final class IterationWrapper implements FinalMoveSelectionStrategy {

    //-------------------------------------------------------------------------

    /**
     * The used FinalMoveSelectionStrategy
     */
    FinalMoveSelectionStrategy finalMoveSelectionStrategy;

    //-------------------------------------------------------------------------

    /**
     * Constructor with FinalMoveSelectionStrategy as input
     *
     * @param finalMoveSelectionStrategy Ludii's FinalMoveSelectionStrategy class
     */
    public IterationWrapper(FinalMoveSelectionStrategy finalMoveSelectionStrategy) {
        this.finalMoveSelectionStrategy = finalMoveSelectionStrategy;
    }

    /**
     * Prints the number of iterations before selecting the final move
     *
     * @param mcts     Ludii's MCTS class
     * @param rootNode Node representing the game position in which the move needs to be selected
     * @return The best move to play according to the FinalMoveSelectionStrategy
     */
    public Move selectMove(MCTS mcts, BaseNode rootNode) {
        System.out.println(mcts.getNumMctsIterations());

        return this.finalMoveSelectionStrategy.selectMove(mcts, rootNode);
    }

    /**
     * Copies the customise method of the initialised FinalMoveSelectionStrategy class
     *
     * @param inputs inputs to customise
     */
    public void customise(String[] inputs) {
        this.finalMoveSelectionStrategy.customise(inputs);
    }
}

