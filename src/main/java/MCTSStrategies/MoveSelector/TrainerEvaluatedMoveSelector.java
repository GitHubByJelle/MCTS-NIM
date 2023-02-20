package MCTSStrategies.MoveSelector;

import Evaluator.NeuralNetworkLeafEvaluator;
import MCTSStrategies.MoveSelector.BatchedEvaluatedMoveSelector;
import game.Game;
import main.collections.FastArrayList;
import other.RankUtils;
import other.context.Context;
import other.context.TempContext;
import other.move.Move;
import utils.TranspositionTableLearning;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Move selector with GameStateEvaluator which can be used during the play-out
 * PLEASE NOTE: All children will be evaluated batched, which only works for leaf evaluators that are based
 * on Neural Networks.
 * The Move selector will additionally save all encountered terminal and non-leaf states (similar as descent).
 */
public class TrainerEvaluatedMoveSelector extends BatchedEvaluatedMoveSelector {

    //-------------------------------------------------------------------------

    /** Transposition table which can be used to store all encountered terminal and non-leaf states */
    TranspositionTableLearning TTTraining;

    //-------------------------------------------------------------------------

    /**
     * Constructor requring no inputs
     */
    public TrainerEvaluatedMoveSelector() {
        super();
    }

    /**
     * Selects the move for the given game position according to the GameStateEvaluators of the leaf and
     * terminal nodes which can be used during play-out. It collects all data to evaluate the children batched, while
     * also saving all terminal and non-leaf nodes.
     *
     * @param context Ludii's context of the current game position
     * @param maybeLegalMoves Moves which can be legal
     * @param p Current player to move
     * @param isMoveReallyLegal Function to check if move is really legal for given game
     * @return Best move according to GameStateEvaluators
     */
    public Move selectMove(final Context context, final FastArrayList<Move> maybeLegalMoves, final int p, final IsMoveReallyLegal isMoveReallyLegal) {
        // Initialise all needed variables
        Game game = context.game();
        List<Move> bestMoves = new ArrayList();
        ArrayList<Integer> nonTerminalMoves = new ArrayList<>();
        float bestValue = Float.NEGATIVE_INFINITY;
        int numLegalMoves = maybeLegalMoves.size();
        final int multiplier = p == 1 ? 1 : -1;
        long zobrist = context.state().fullHash(context);

        // For all legal moves
        Move move;
        boolean terminal = false;
        for (int i = 0; i < numLegalMoves; i++) {
            move = maybeLegalMoves.get(i);
            if (isMoveReallyLegal.checkMove(move)) {
                TempContext copyContext = new TempContext(context);
                game.apply(copyContext, move);
                float heuristicScore;

                // Evaluate the terminal states, while saving the index of the non-terminal states
                if (!copyContext.trial().over() && copyContext.active(p)) {
                    heuristicScore = Float.NEGATIVE_INFINITY;
                    nonTerminalMoves.add(i);
                } else {
                    heuristicScore = this.terminalStateEvaluator.evaluate(copyContext, p);
                    terminal = true;

                    // Store all terminal states in TT (similar to descent)
                    long zobristChild = copyContext.state().fullHash(copyContext);
                    boolean nullValue = this.TTTraining.getEntry(zobristChild) == null;
                    if (nullValue){
                        this.TTTraining.store(zobristChild, heuristicScore * multiplier, 999,
                                ((NeuralNetworkLeafEvaluator) leafEvaluator).boardToInput(copyContext).data().asFloat());
                    }
                    else {
                        synchronized (this.TTTraining.getEntry(zobristChild)){
                            this.TTTraining.store(zobristChild, heuristicScore * multiplier, 999,
                                    ((NeuralNetworkLeafEvaluator) leafEvaluator).boardToInput(copyContext).data().asFloat());
                        }
                    }
                }

                // If a terminal node is found, add this to the best moves
                if (heuristicScore > bestValue) {
                    bestValue = heuristicScore;
                    bestMoves.clear();
                    bestMoves.add(move);
                } else if (heuristicScore == bestValue) {
                    bestMoves.add(move);
                }
            }
        }

        if (terminal) {
            // Store best Value of context in TT (if terminal)
            boolean nullValue = this.TTTraining.getEntry(zobrist) == null;
            if (nullValue){
                this.TTTraining.store(zobrist, bestValue * multiplier, 999,
                        ((NeuralNetworkLeafEvaluator) leafEvaluator).boardToInput(context).data().asFloat());
            }
            else {
                synchronized (this.TTTraining.getEntry(zobrist)){
                    this.TTTraining.store(zobrist, bestValue * multiplier, 999,
                            ((NeuralNetworkLeafEvaluator) leafEvaluator).boardToInput(context).data().asFloat());
                }
            }
            return bestMoves.get(ThreadLocalRandom.current().nextInt(bestMoves.size()));
        }

        // Evaluate all non-terminal nodes (leaves) together (batched)
        float[] nonTerminalMoveScores = ((NeuralNetworkLeafEvaluator) leafEvaluator).evaluateMoves(context,
                nonTerminalMoves, p);

        // Determine best move
        for (int i = 0; i < nonTerminalMoves.size(); i++) {
            float heuristicScore = nonTerminalMoveScores[i];
            if (heuristicScore > bestValue) {
                bestValue = heuristicScore;
                bestMoves.clear();
                bestMoves.add(maybeLegalMoves.get(nonTerminalMoves.get(i)));
            } else if (heuristicScore == bestValue) {
                bestMoves.add(maybeLegalMoves.get(nonTerminalMoves.get(i)));
            }
        }

        // Store bestValue found after search in TT
        boolean nullValue = this.TTTraining.getEntry(zobrist) == null;
        if (nullValue){
            this.TTTraining.store(zobrist, bestValue * multiplier, 999,
                    ((NeuralNetworkLeafEvaluator) leafEvaluator).boardToInput(context).data().asFloat());
        }
        else {
            synchronized (this.TTTraining.getEntry(zobrist)){
                this.TTTraining.store(zobrist, bestValue * multiplier, 999,
                        ((NeuralNetworkLeafEvaluator) leafEvaluator).boardToInput(context).data().asFloat());
            }
        }

        return bestMoves.get(ThreadLocalRandom.current().nextInt(bestMoves.size()));
    }

    /**
     * Setter for the Transposition Table used for Training
     *
     * @param TTTraining Transposition table which can be used to store all encountered terminal and non-leaf states
     */
    public void setTTTraining(TranspositionTableLearning TTTraining) {
        this.TTTraining = TTTraining;
    }
}