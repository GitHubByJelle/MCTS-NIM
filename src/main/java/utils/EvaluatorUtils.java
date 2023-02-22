package utils;

import Evaluator.GameStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import other.context.Context;
import other.move.Move;

import java.util.ArrayList;

/**
 * Util to easily evaluate children batched or single based on given GameStateEvaluators.
 */
public class EvaluatorUtils {
    /**
     * Evaluate children single (non-batched)
     *
     * @param context                Ludii's context
     * @param legalMoves             All legal move in current game state
     * @param mover                  Current mover
     * @param leafEvaluator          GameStateEvaluator for leafNodes
     * @param terminalStateEvaluator GameStateEvaluators for terminal game positions
     * @return Values indicating how good the game state is for all legal moves (higher is better).
     */
    public static double[] EvaluateChildren(Context context, Move[] legalMoves, int mover,
                                            GameStateEvaluator leafEvaluator, GameStateEvaluator terminalStateEvaluator) {
        // Evaluate all children
        double value;
        int numLegalMoves = legalMoves.length;
        double[] result = new double[numLegalMoves];
        for (int i = 0; i < numLegalMoves; i++) {
            Context contextCopy = new Context(context);
            contextCopy.game().apply(contextCopy, legalMoves[i]);

            if (contextCopy.trial().over()) {
                value = terminalStateEvaluator.evaluate(contextCopy, mover);
            } else {
                value = leafEvaluator.evaluate(contextCopy, mover);
            }

            result[i] = value;
        }

        return result;
    }

    /**
     * Evaluate children batched. Useful for NNs.
     *
     * @param context                Ludii's context
     * @param legalMoves             All legal move in current game state
     * @param mover                  Current mover
     * @param leafEvaluator          NeuralNetworkLeafEvaluator for leafNodes
     * @param terminalStateEvaluator GameStateEvaluators for terminal game positions
     * @return Values indicating how good the game state is for all legal moves (higher is better).
     */
    public static double[] EvaluateChildrenBatched(Context context, Move[] legalMoves, int mover,
                                                   NeuralNetworkLeafEvaluator leafEvaluator,
                                                   GameStateEvaluator terminalStateEvaluator) {
        // Evaluate all children
        double value;
        int numLegalMoves = legalMoves.length;
        ArrayList<Integer> nonTerminalMoves = new ArrayList<>();
        double[] result = new double[numLegalMoves];
        // For all children evaluate terminal, otherwise add to non terminal
        for (int i = 0; i < numLegalMoves; i++) {
            Context contextCopy = new Context(context);
            contextCopy.game().apply(contextCopy, legalMoves[i]);

            if (contextCopy.trial().over()) {
                value = terminalStateEvaluator.evaluate(contextCopy, mover);
                result[i] = value;
            } else {
                nonTerminalMoves.add(i);
            }
        }

        // Evaluate all non-terminal nodes (leaves) together (batched)
        if (nonTerminalMoves.size() > 0) {
            float[] nonTerminalMoveScores = leafEvaluator.evaluateMoves(context,
                    nonTerminalMoves, mover);
            for (int i = 0; i < nonTerminalMoves.size(); i++) {
                result[nonTerminalMoves.get(i)] = nonTerminalMoveScores[i];
            }
        }

        return result;
    }
}
