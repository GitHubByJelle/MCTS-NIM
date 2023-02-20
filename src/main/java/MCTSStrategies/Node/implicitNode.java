package MCTSStrategies.Node;

import Evaluator.GameStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import other.context.Context;
import other.move.Move;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;
import search.mcts.nodes.DeterministicNode;
import utils.EvaluatorUtils;

import java.util.Arrays;

/**
 * Node class which can be used in the Ludii's MCTS class, which is needed for an implicit MCTS implementation.
 * Compared to the normal nodes, it stores the best estimated value and the initial values, that can be used in the
 * UCT function.
 * Backpropagates values as proposed in:
 * Lanctot, M., Winands, M. H., Pepels, T., & Sturtevant, N. R. (2014, August). Monte Carlo tree search with
 * heuristic evaluations using implicit minimax backups. In 2014 IEEE Conference on Computational Intelligence and
 * Games (pp. 1-8). IEEE.
 */
public class implicitNode extends DeterministicNode {

    //-------------------------------------------------------------------------

    /** Stores the index of the best estimated value based on a leaf evaluator (of the GameStatEvaluator class) */
    protected int bestEstimatedIndex = -1;

    /** Stores the best estimated value based on a leaf evaluator (of the GameStatEvaluator class) */
    protected double bestEstimatedValue = -9999999;

    /** Stores the initial estimated value of all children based on a leaf evaluator (of the GameStatEvaluator class) */
    protected double[] initialEstimatedValues = null;

    //-------------------------------------------------------------------------

    /**
     * Constructor to create a node which can be used for implicit MCTS implementations
     *
     * @param mcts Ludii's MCTS class
     * @param parent Parent node of current node
     * @param parentMove Node from parent to current node
     * @param parentMoveWithoutConseq Node from parent to current node
     * @param context Ludii's context class representating the game state
     * @param leafEvaluator GameStateEvaluator which can be used to evaluate non-terminal game states
     * @param terminalStateEvaluator GameStateEvaluator which can be used to evaluate terminal game states
     * @param evaluateBatched Indicates if the leaf evaluator should calculate the children batched (useful for NNs)
     */
    public implicitNode(MCTS mcts, BaseNode parent, Move parentMove, Move parentMoveWithoutConseq, Context context,
                        GameStateEvaluator leafEvaluator, GameStateEvaluator terminalStateEvaluator,
                        boolean evaluateBatched) {
        // Initialise node
        super(mcts, parent, parentMove, parentMoveWithoutConseq, context);
        final int mover = context.state().playerToAgent(context.state().mover());

        // If game isn't over
        if (!context.trial().over()) {

            // Evaluate all children
            this.initialEstimatedValues = this.evaluateChildren(leafEvaluator, terminalStateEvaluator,
                    evaluateBatched, mover);

            // Get the best child
            double value;
            for (int i = 0; i < this.numLegalMoves(); i++) {
                value = this.initialEstimatedValues[i];
                if (value > this.bestEstimatedValue){
                    this.bestEstimatedValue = value;
                    this.bestEstimatedIndex = i;
                }
            }
        }
        // Else evaluate current state
        else {
            this.bestEstimatedValue = terminalStateEvaluator.evaluate(context, mover);
        }

        // If a parent exist, update the best estimated value (if needed) in a minimax fashion
        if (this.parent != null){
            ((implicitNode)this.parent).implicitMinimaxBackup(this);
        }
    }

    /**
     * Updates the best estimated value of a node using the minimax framework, iff it is required to update the value.
     * The value if updated when a better best value has been found, or the best child received a new value.
     * @param fromChild Child node with the updated values
     */
    protected void implicitMinimaxBackup(implicitNode fromChild){
        // Take negative value (if mover changed) and check if best child got changed
        double backupValue = this.context.state().playerToAgent(this.context.state().mover()) ==
                fromChild.context.state().playerToAgent(fromChild.context.state().mover()) ?
                fromChild.bestEstimatedValue : -fromChild.bestEstimatedValue;
        boolean bestChild = this.nthLegalMove(this.bestEstimatedIndex).equals(fromChild.parentMove);

        // Debugging:
//        if (this.parent == null){
//            System.out.printf("Start values: [");
//            for (int i = 0; i < this.numLegalMoves(); i++){
//                if (this.children[i] == null){
//                    System.out.printf("%.4f, ", this.initialHeuristicValues[i]);
//                }
//                else {
//                    System.out.printf("%.4f, ",
//                            this.context.state().playerToAgent(this.context.state().mover()) ==
//                                    this.children[i].contextRef().state().playerToAgent(
//                                            this.children[i].contextRef().state().mover()) ?
//                                    ((implicitNode)this.children[i]).bestHeuristicValue :
//                                    -((implicitNode)this.children[i]).bestHeuristicValue);
//                }
//            }
//            System.out.println("]");
//            System.out.println("bestMove: " + this.legalMoves[this.bestHeuristicIndex]);
//            System.out.println("back up move: " + fromChild.parentMove);
//            System.out.println("Is best child? " + bestChild);
//            System.out.println("Back up value: " + backupValue);
//        }

//        System.out.println(this.legalMoves[this.bestHeuristicIndex] + " " + fromChild.parentMove + " " + bestChild);

        // If not current best child, but better value, change to new value
        if (!bestChild && backupValue > this.bestEstimatedValue){
            synchronized (this){
                // Set best value
                this.bestEstimatedValue = backupValue;

                // Find index of new best move
                for (int i = 0; i < this.numLegalMoves(); i++) {
                    if (this.legalMoves[i].hashCode() == fromChild.parentMove.hashCode()){
                        this.bestEstimatedIndex = i;
                        break;
                    }
                }

                // Update parent
                if (this.parent != null){
                    ((implicitNode)this.parent).implicitMinimaxBackup(this);
                }
            }
        }
        // If best index got changed, look at all moves
        else if (bestChild) {
            synchronized (this){
                double bestValue = Double.NEGATIVE_INFINITY;
                int bestIndex = -1;
                double value;
                for (int i = 0; i < this.numLegalMoves(); i++) {
                    // If the child is new, the child hasn't been added to the parent
                    // So the child will be null. To prevent this from happening, check if index matches
                    if (i == this.bestEstimatedIndex) {
                        value = backupValue;
                    }
                    else if (this.children[i] ==  null){
                        value = this.initialEstimatedValues[i];
                    }
                    else {
                        value = this.context.state().playerToAgent(this.context.state().mover()) ==
                                this.children[i].contextRef().state().playerToAgent(
                                        this.children[i].contextRef().state().mover()) ?
                                ((implicitNode)this.children[i]).bestEstimatedValue :
                                -((implicitNode)this.children[i]).bestEstimatedValue;
                    }

                    if (value > bestValue){
                        bestValue = value;
                        bestIndex = i;
                    }
                }

                // Save best value and index
                this.bestEstimatedValue = bestValue;
                this.bestEstimatedIndex = bestIndex;

                // Update parent
                if (this.parent != null){
                    ((implicitNode)this.parent).implicitMinimaxBackup(this);
                }
            }
        }

        // Debugging: check if the value got actually changed
//        if (this.parent == null){
//            System.out.printf("Best value of node: %.4f\n",this.bestHeuristicValue);
//            System.out.printf("Best value index: %d\n", this.bestHeuristicIndex);
//            System.out.println("###############\n");
//        }

        // If nothing got changed, don't continue minimax backup
    }

    /**
     * Evaluates the children based on the leaf and terminal evaluator
     *
     * @param leafEvaluator GameStateEvaluator which can be used to evaluate non-terminal game states
     * @param terminalStateEvaluator GameStateEvaluator which can be used to evaluate terminal game states
     * @param evaluateBatched Indicates if the leaf evaluator should calculate the children batched (useful for NNs)
     * @param mover ID of the player to move
     * @return An array with the evaluation of all children of the current game position
     */
    protected double[] evaluateChildren(GameStateEvaluator leafEvaluator, GameStateEvaluator terminalStateEvaluator,
                                        boolean evaluateBatched, int mover){
        double[] result;
        if (evaluateBatched){
            result = EvaluatorUtils.EvaluateChildrenBatched(this.context, this.legalMoves, mover,
                    (NeuralNetworkLeafEvaluator) leafEvaluator, terminalStateEvaluator);
        }
        else {
            result = EvaluatorUtils.EvaluateChildren(this.context, this.legalMoves, mover, leafEvaluator,
                    terminalStateEvaluator);
        }

        return result;
    }

    // PLEASE NOTE: VLoss Average gives the possibility to also ADD score if it is losing. Is this what you want?
//    @Override
//    public double expectedScore(int agent) {
//        return this.numVisits == 0 ? 0.0 : (this.totalScores[agent] - (double)this.numVirtualVisits.get() * this.totalScores[agent] / (double)this.numVisits) / (double)(this.numVisits + this.numVirtualVisits.get());
//    }

//    @Override
//    public double exploitationScore(int agent) {
//        return this.numVisits == 0 ? 0.0 : (this.totalScores[agent] - (double)this.numVirtualVisits.get()) / (double)(this.numVisits + this.numVirtualVisits.get());
//    }

//    @Override
//    public double expectedScore(int agent) {
//        return this.numVisits == 0 ? 0.0 : (this.totalScores[agent]) / (double)(this.numVisits);
//    }

    /**
     * Converts the current node to a string
     * @return String providing information of the current node
     */
    @Override
    public String toString() {
        final int mover = this.contextRef().state().playerToAgent(this.contextRef().state().mover()) == 1 ? 2 : 1;
        if (this == null){
            return "unvisited";
        }

        return String.format("from %d, to %d. Scores: %s, numVisits: %d, win rate: %.4f, bestHeuristic: %.4f",
                this.parentMove.from(), this.parentMove.to(),
                Arrays.toString(this.totalScores), this.numVisits,
                this.exploitationScore(mover),
                this.bestEstimatedValue);
    }

    /**
     * Getter for the best estimated value
     * @return the best estimated value
     */
    public double getBestEstimatedValue() {
        return bestEstimatedValue;
    }

    /**
     * Getter for the initial estimated value of a specific child
     *
     * @param index The index of the child
     * @return initial estimated value of child at the given index
     */
    public double getInitialEstimatedValue(int index) {
        return initialEstimatedValues[index];
    }
}
