package MCTSStrategies.Wrapper;

import Evaluator.NeuralNetworkLeafEvaluator;
import MCTSStrategies.Node.implicitNode;
import game.Game;
import other.context.Context;
import other.move.Move;
import search.mcts.MCTS;
import search.mcts.finalmoveselection.FinalMoveSelectionStrategy;
import search.mcts.nodes.BaseNode;
import utils.Enums;
import utils.TranspositionTableLearning;

/**
 * Wrapper class for FinalMoveSelectionStrategy which stores all values made during the selection
 * into a Transposition Table, such that they can be used during Training
 */
public class TrainingSelectionWrapper implements FinalMoveSelectionStrategy {

    //-------------------------------------------------------------------------

    /**
     * The used FinalMoveSelectionStrategy
     */
    FinalMoveSelectionStrategy finalMoveSelectionStrategy;

    /**
     * Transposition Table used to store all training data
     */
    TranspositionTableLearning TTTraining = new TranspositionTableLearning(12);

    /**
     * GameStateEvaluator which can be used to evaluate non-terminal game states
     */
    NeuralNetworkLeafEvaluator leafEvaluator;

    /**
     * Enum for the data selection strategy used from the descent framework as proposed in:
     * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
     * knowledge. arXiv preprint arXiv:2008.01188.
     */
    Enums.DataSelection dataSelection;

    /**
     * Constructor with FinalMoveSelectionStrategy as input
     *
     * @param finalMoveSelectionStrategy Ludii's FinalMoveSelectionStrategy class
     */
    public TrainingSelectionWrapper(FinalMoveSelectionStrategy finalMoveSelectionStrategy) {
        this.finalMoveSelectionStrategy = finalMoveSelectionStrategy;
    }

    /**
     * Stores all selected nodes (based on the data selection strategy) to
     * the Transposition Table before returning the move
     *
     * @param mcts     Ludii's MCTS class
     * @param rootNode Node representing the game position in which the move needs to be selected
     * @return The best move to play according to the FinalMoveSelectionStrategy
     */
    public Move selectMove(MCTS mcts, BaseNode rootNode) {
        // If tree learning, store all children
        if (this.dataSelection == Enums.DataSelection.TREE) {
            this.saveAllChildrenToTT((implicitNode) rootNode, 0);
        }
        // Else, only store root node
        else {
            this.storeNode((implicitNode) rootNode, 0);
        }

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

    /**
     * Saves all children belonging to the given node to recursively to the Transposition Table. All children of
     * all children also get added. Since this is based on tree learning from the descent framework, non-terminal leaf
     * nodes don't get added to the transposition table.
     *
     * @param node  Node representing the game position that needs to be added to the transposition table
     * @param depth Depth of the current node
     */
    public void saveAllChildrenToTT(implicitNode node, int depth) {
        int numChildren = node.numLegalMoves();
        for (int i = 0; i < numChildren; i++) {
            implicitNode child = (implicitNode) node.childForNthLegalMove(i);
            if (child != null) {
                // Store the node in the TT
                this.storeNode(child, depth);

                // Save all his children
                this.saveAllChildrenToTT(child, depth + 1);
            }
        }
    }

    /**
     * Stores the data of a single node to the transposition table
     *
     * @param node  Node representing the game position that needs to be added to the transposition table
     * @param depth Depth of the current node
     */
    public void storeNode(implicitNode node, int depth) {
        final Context context = node.contextRef();
        final int mover = context.state().playerToAgent(context.state().mover());
        final int multiplier = mover == 1 ? 1 : -1;
        final long zobrist = context.state().fullHash(context);
        this.TTTraining.store(zobrist, (float) node.getBestEstimatedValue() * multiplier, depth,
                this.leafEvaluator.boardToInput(context).data().asFloat());
    }

    /**
     * Creates a NeuralNetworkLeafEvaluator which is used to convert the to NN input. Please note, the network
     * doesn't have a NeuralNetwork to perform evaluations with!
     *
     * @param game Ludii's game class
     */
    public void createLeafEvaluator(Game game) {
        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, null);
    }

    /**
     * Setter for the transposition table used to store the learning data
     *
     * @param TTTraining The transposition table used to store the learning data
     */
    public void setTTTraining(TranspositionTableLearning TTTraining) {
        this.TTTraining = TTTraining;
    }

    /**
     * Setter for the data selection strategy used
     *
     * @param dataSelection The data selection strategy used
     */
    public void setDataSelection(Enums.DataSelection dataSelection) {
        this.dataSelection = dataSelection;
    }
}
