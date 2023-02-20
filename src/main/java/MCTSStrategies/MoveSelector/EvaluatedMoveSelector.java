package MCTSStrategies.MoveSelector;

import Evaluator.GameStateEvaluator;
import game.Game;
import main.collections.FastArrayList;
import metadata.ai.heuristics.Heuristics;
import other.RankUtils;
import other.context.Context;
import other.context.TempContext;
import other.move.Move;
import other.playout.PlayoutMoveSelector;
import utils.Enums;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Move selector with GameStateEvaluator which can be used during the play-out
 * PLEASE NOTE: All children will be evaluated individually, which is not optimal for NNs.
 */
public class EvaluatedMoveSelector extends PlayoutMoveSelector {

    //-------------------------------------------------------------------------

    /** GameStatEvaluator to evaluate leaf nodes */
    protected GameStateEvaluator leafEvaluator = null;

    /** GameStatEvaluator to evaluate terminal nodes */
    protected GameStateEvaluator terminalStateEvaluator = null;

    //-------------------------------------------------------------------------

    /**
     * Constructor requring no inputs
     */
    public EvaluatedMoveSelector() {
    }

    /**
     * Selects the move for the given game position according to the GameStateEvaluators of the leaf and
     * terminal nodes which can be used during play-out
     *
     * @param context Ludii's context of the current game position
     * @param maybeLegalMoves Moves which can be legal
     * @param p Current player to move
     * @param isMoveReallyLegal Function to check if move is really legal for given game
     * @return Best move according to GameStateEvaluators
     */
    public Move selectMove(final Context context, final FastArrayList<Move> maybeLegalMoves, final int p, final PlayoutMoveSelector.IsMoveReallyLegal isMoveReallyLegal) {
        // Initialise all needed variables
        Game game = context.game();
        List<Move> bestMoves = new ArrayList();
        float bestValue = Float.NEGATIVE_INFINITY;
        Iterator iterator = maybeLegalMoves.iterator();

        // While a next legal move exist, keep going
        while(true) {
            Move move;
            do {
                // If all legal moves have been seen, return random best move with randomness as tie-breaker
                if (!iterator.hasNext()) {
                    if (bestMoves.size() > 0) {
                        return (Move)bestMoves.get(ThreadLocalRandom.current().nextInt(bestMoves.size()));
                    }

                    return null;
                }

                move = (Move)iterator.next();
            } while(!isMoveReallyLegal.checkMove(move));

            // If the move is legal, evaluate the childs
            TempContext copyContext = new TempContext(context);
            game.apply(copyContext, move);
            float heuristicScore = 0.0F;
            if (!copyContext.trial().over() && copyContext.active(p)) {
                heuristicScore = this.leafEvaluator.evaluate(copyContext,p);
            } else {
                heuristicScore = this.terminalStateEvaluator.evaluate(copyContext,p);
            }

            // If the score is better than the best value found, update best move
            if (heuristicScore > bestValue) {
                bestValue = heuristicScore;
                bestMoves.clear();
                bestMoves.add(move);
            } else if (heuristicScore == bestValue) {
                bestMoves.add(move);
            }
        }
    }

    /**
     * Setter for the leaf evaluator used during playout
     *
     * @param leafEvaluator GameStateEvaluator for the leaf nodes
     */
    public void setLeafEvaluator(GameStateEvaluator leafEvaluator) {
        this.leafEvaluator = leafEvaluator;
    }

    /**
     * Setter for the terminal state evaluator used during playout
     *
     * @param terminalStateEvaluator GameStateEvaluator for the terminal nodes
     */
    public void setTerminalStateEvaluator(GameStateEvaluator terminalStateEvaluator) {
        this.terminalStateEvaluator = terminalStateEvaluator;
    }
}