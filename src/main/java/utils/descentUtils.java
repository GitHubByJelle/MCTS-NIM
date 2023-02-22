package utils;

import other.move.Move;
import utils.data_structures.ScoredMove;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class descentUtils {
    /**
     * Add to new backpropagated value to the sorted scored move list by only repositioning the explored
     * scored move.
     *
     * @param move              The move made
     * @param outputScore       The new score of the move
     * @param nbV               The number of visitis of the updates move
     * @param sortedScoredMoves List with sorted scored moves
     * @param bestIndex         Index of the current best move
     * @param numLegalMoves     Number of legal moves
     * @param mover             The player to move
     * @param maximisingPlayer  The maxisimising player (always player one for UBFM)
     * @return Newly sorted list with all scored moves
     */
    public static List<ScoredMove> addScoreToSortedScoredMoves(Move move, float outputScore, int nbV,
                                                               List<ScoredMove> sortedScoredMoves,
                                                               int bestIndex, int numLegalMoves,
                                                               int mover, int maximisingPlayer) {
        // Finding correct spot in sortedScored moves to place the new move (start by copying from current spot
        // of move).
        int k = bestIndex;
        label99:
        while (true) {
            while (true) {
                // If out of index stop
                if (k >= numLegalMoves - 1) {
                    break label99;
                }

                // If next scored move is better than most promising move, copy next Scoredmove
                if (((ScoredMove) sortedScoredMoves.get(k + 1)).score >= outputScore && mover == maximisingPlayer || ((ScoredMove) sortedScoredMoves.get(k + 1)).score <= outputScore && mover != maximisingPlayer) {
                    sortedScoredMoves.set(k, (ScoredMove) sortedScoredMoves.get(k + 1));
                    ++k;
                } else {
                    // If out of index, or
                    // (If previous scored move is higher than most promising move or mover is minimizing, and
                    // If previous scored move is lower than most promising move or mover is maximizing)
                    // Stop in total
                    if (k <= 0 || (!(((ScoredMove) sortedScoredMoves.get(k - 1)).score < outputScore) || mover != maximisingPlayer) && (!(((ScoredMove) sortedScoredMoves.get(k - 1)).score > outputScore) || mover == maximisingPlayer)) {
                        break label99;
                    }

                    sortedScoredMoves.set(k, (ScoredMove) sortedScoredMoves.get(k - 1));
                    --k;
                }
            }
        }

        // Set new value at correct position
        sortedScoredMoves.set(k, new ScoredMove(move, outputScore, nbV + 1));

        // Return re-ordered sortedScoredMoves
        return sortedScoredMoves;
    }

    /**
     * Get the best action from the sorted scored moves list based on the selected exploration policy
     *
     * @param scoredMoves   List with sorted scored moves
     * @param numLegalMoves Number of legal moves in current game position
     * @return Index of selected action
     */
    public static int getBestAction(List<ScoredMove> scoredMoves, int numLegalMoves,
                                    Enums.ExplorationPolicy explorationPolicy,
                                    double explorationEpsilon) {
        int indexPicked;
        switch (explorationPolicy) {
            case BEST:
                indexPicked = 0;
                break;
            case EPSILON_GREEDY:
                if (ThreadLocalRandom.current().nextDouble(1.0) < explorationEpsilon) {
                    indexPicked = ThreadLocalRandom.current().nextInt(numLegalMoves);
                } else {
                    indexPicked = 0;
                }
                break;
            case SOFTMAX:
                // Determine exponential sum and probabilities
                double expSum = IntStream.range(0, numLegalMoves).mapToDouble((x) ->
                        Math.exp(scoredMoves.get(x).score)).sum();
                double[] probs = IntStream.range(0, numLegalMoves).mapToDouble((x) ->
                        (Math.exp(scoredMoves.get(x).score) / expSum)).toArray();

                // Take random sample
                indexPicked = -1;
                double rand = ThreadLocalRandom.current().nextDouble(1.0);
                for (int i = 0; i < probs.length; ++i) {
                    rand -= probs[i];
                    if (rand <= 0) {
                        indexPicked = i;
                        break;
                    }
                }
                break;
            default:
                throw new RuntimeException("Unkown exploration policy");
        }

        return indexPicked;
    }

    /**
     * Get the best action to play in the actual game based on the selection policy used
     *
     * @param rootTableData Table from the root node in the transposition table
     * @param maximising    Indicates if the player is maximising
     * @return The best scored move according to the selection policy
     */
    public static ScoredMove finalMoveSelection(TranspositionTableStamp.StampTTData rootTableData,
                                                Enums.SelectionPolicy selectionPolicy, boolean maximising) {
        switch (selectionPolicy) {
            case BEST:
                return (ScoredMove) rootTableData.sortedScoredMoves.get(0);
            case SAFEST:
                ScoredMove scoredMove;
                ScoredMove safestScoredMove = (ScoredMove) rootTableData.sortedScoredMoves.get(0);

                for (int i = 0; i < rootTableData.sortedScoredMoves.size(); ++i) {
                    scoredMove = (ScoredMove) rootTableData.sortedScoredMoves.get(i);
                    if (scoredMove.nbVisits > safestScoredMove.nbVisits || scoredMove.nbVisits == safestScoredMove.nbVisits && (maximising && scoredMove.score > safestScoredMove.score || !maximising && scoredMove.score < safestScoredMove.score)) {
                        safestScoredMove = scoredMove;
                    }
                }

                return safestScoredMove;
            default:
                System.err.println("Error: selectionPolicy not implemented");
                return (ScoredMove) rootTableData.sortedScoredMoves.get(0);
        }
    }

    /**
     * Get the best action to play in the actual game based on the selection policy used
     *
     * @param rootTableData Table from the root node in the transposition table
     * @param maximising    Indicates if the player is maximising
     * @return The best scored move according to the selection policy
     */
    public static CompletedMove finalMoveSelection(TranspositionTableStampCompleted.StampTTDataCompleted rootTableData,
                                                Enums.SelectionPolicy selectionPolicy, boolean maximising) {
        switch (selectionPolicy) {
            case BEST:
                return rootTableData.sortedScoredMoves.get(0);
            case SAFEST:
                CompletedMove scoredMove;
                CompletedMove safestScoredMove = rootTableData.sortedScoredMoves.get(0);

                for (int i = 0; i < rootTableData.sortedScoredMoves.size(); ++i) {
                    scoredMove = rootTableData.sortedScoredMoves.get(i);
                    if (scoredMove.nbVisits > safestScoredMove.nbVisits || scoredMove.nbVisits == safestScoredMove.nbVisits && (maximising && scoredMove.score > safestScoredMove.score || !maximising && scoredMove.score < safestScoredMove.score)) {
                        safestScoredMove = scoredMove;
                    }
                }

                return safestScoredMove;
            default:
                System.err.println("Error: selectionPolicy not implemented");
                return rootTableData.sortedScoredMoves.get(0);
        }
    }
}
