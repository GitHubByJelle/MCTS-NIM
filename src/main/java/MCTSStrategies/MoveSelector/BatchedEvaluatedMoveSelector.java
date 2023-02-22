package MCTSStrategies.MoveSelector;

import Evaluator.NeuralNetworkLeafEvaluator;
import game.Game;
import main.collections.FastArrayList;
import other.context.Context;
import other.context.TempContext;
import other.move.Move;
import other.playout.PlayoutMoveSelector;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Move selector with GameStateEvaluator which can be used during the play-out
 * PLEASE NOTE: All children will be evaluated batched, which only works for leaf evaluators that are based
 * on Neural Networks.
 */
public class BatchedEvaluatedMoveSelector extends EvaluatedMoveSelector {

    /**
     * Constructor requring no inputs
     */
    public BatchedEvaluatedMoveSelector() {
        super();
    }

    /**
     * Selects the move for the given game position according to the GameStateEvaluators of the leaf and
     * terminal nodes which can be used during play-out. It collects all data to evaluate the children batched.
     *
     * @param context           Ludii's context of the current game position
     * @param maybeLegalMoves   Moves which can be legal
     * @param p                 Current player to move
     * @param isMoveReallyLegal Function to check if move is really legal for given game
     * @return Best move according to GameStateEvaluators
     */
    public Move selectMove(final Context context, final FastArrayList<Move> maybeLegalMoves, final int p, final PlayoutMoveSelector.IsMoveReallyLegal isMoveReallyLegal) {
        // Initialise all needed variables
        Game game = context.game();
        List<Move> bestMoves = new ArrayList();
        ArrayList<Integer> nonTerminalMoves = new ArrayList<>();
        float bestValue = Float.NEGATIVE_INFINITY;
        int numLegalMoves = maybeLegalMoves.size();

        // For all legal moves
        Move move;
        boolean terminal = false;
        for (int i = 0; i < numLegalMoves; i++) {
            move = maybeLegalMoves.get(i);
            if (isMoveReallyLegal.checkMove(move)) {
                TempContext copyContext = new TempContext(context);
                game.apply(copyContext, move);
                float heuristicScore = 0.0F;
                // Evaluate the terminal states, while saving the index of the non-terminal states
                if (!copyContext.trial().over() && copyContext.active(p)) {
                    heuristicScore = Float.NEGATIVE_INFINITY;
                    nonTerminalMoves.add(i);
                } else {
                    heuristicScore = this.terminalStateEvaluator.evaluate(copyContext, p);
                    terminal = true;
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

        // If a terminal move is found, execute this move
        if (terminal) {
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

        return bestMoves.get(ThreadLocalRandom.current().nextInt(bestMoves.size()));
    }
}