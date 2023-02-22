package MCTSStrategies.Wrapper;

import other.move.Move;
import search.mcts.MCTS;
import search.mcts.finalmoveselection.FinalMoveSelectionStrategy;
import search.mcts.nodes.BaseNode;

import java.util.Arrays;

/**
 * Wrapper class for FinalMoveSelectionStrategy which prints values from the rootnode and children, such that
 * these can be used for debugging.
 */
public final class debugFinalSelectionWrapper implements FinalMoveSelectionStrategy {

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
    public debugFinalSelectionWrapper(FinalMoveSelectionStrategy finalMoveSelectionStrategy) {
        this.finalMoveSelectionStrategy = finalMoveSelectionStrategy;
    }

    /**
     * Prints the move selected, values of the root node, and values of the children nodes
     * before selecting the final move
     *
     * @param mcts     Ludii's MCTS class
     * @param rootNode Node representing the game position in which the move needs to be selected
     * @return The best move to play according to the FinalMoveSelectionStrategy
     */
    public Move selectMove(MCTS mcts, BaseNode rootNode) {
        // Print the selected moves
        System.out.println("Selecting best move");
        Move bestMove = this.finalMoveSelectionStrategy.selectMove(mcts, rootNode);

        // Print values of root node
        System.out.printf("Root node: Scores: %s, numVisits: %d, win rate: %.4f\n",
                Arrays.toString(rootNode.totalScores()), rootNode.numVisits(),
                rootNode.expectedScore(rootNode.contextRef().state().playerToAgent(rootNode.contextRef().state().mover())));

        // Print values of all children
        System.out.println("Children:");
        int selected = -1;
        double value = -100;
        int visits = -100;
        for (int i = 0; i < rootNode.numLegalMoves(); i++) {
            final BaseNode child = rootNode.childForNthLegalMove(i);

            System.out.printf((i + 1) + ") " + child);

            // If selected, add selected behind the print, else nothing
            if (rootNode.nthLegalMove(i).equals(bestMove)) {
                System.out.print(" (selected)\n");
                selected = (i + 1);
                value = child.expectedScore(bestMove.mover());
                visits = child.numVisits();
            } else {
                System.out.print("\n");
            }
        }

        // Print the results from the root node
        System.out.println("Rootnode selected: " + selected +
                String.format(", value: %.4f, numVisits: %d", value, visits));

        System.out.println();

        return bestMove;
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

