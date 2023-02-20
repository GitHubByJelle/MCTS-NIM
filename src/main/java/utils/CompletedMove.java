package utils;

import org.jetbrains.annotations.NotNull;
import other.move.Move;
import utils.data_structures.ScoredMove;
import utils.Enums.ExplorationPolicy;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Custom move which is used in the completed UBFM and completed descent implementations as proposed in:
 * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
 * knowledge. arXiv preprint arXiv:2008.01188.
 *
 * Based on implementation from Ludii
 */
public class CompletedMove implements Comparable<CompletedMove> {
    public final Move move;
    public final float score;
    public final int nbVisits;
    public final float completion;
    public final float resolution;

    /**
     * Constructor requiring the move, resolution, completion, score and number of visits
     *
     * @param move The move that has been made
     * @param resolution The resolution (is the game position solved or not)
     * @param completion The completion (is the game a loss, draw or win? (-1, 0, 1, respectively))
     * @param score The score of the given move (based on the GameStateEvaluator)
     * @param nbVisits The number of times this moves has been visited
     */
    public CompletedMove(Move move, float resolution, float completion, float score, int nbVisits) {
        this.move = move;
        this.resolution = resolution;
        this.completion = completion;
        this.score = score;
        this.nbVisits = nbVisits;
    }

    /**
     * Getter for the completion value
     * @return Completion value
     */
    public float getCompletion() {
        return completion;
    }

    /**
     * Getter for the score
     * @return Score
     */
    public float getScore() {
        return score;
    }

    /**
     * Getter for the number of visits
     * @return the number of visits
     */
    public int getNbVisits(){
        return nbVisits;
    }

    /**
     * Getter for negative number of visits
     * @return the number of visits multiplied by -1
     */
    public int getNegativeNbVisits(){
        return -nbVisits;
    }

    /**
     * Compare function to see which CompletedMove performs better as described by Cohen-Solal.
     * The order is based on Completion, value, number of visits
     *
     * @param other the CompletedMove to be compared.
     * @return
     */
    public int compareTo(CompletedMove other) {
        float deltaC = other.completion - this.completion;
        if (deltaC < 0.0F) {
            return -1;
        } else if (deltaC > 0.0F){
            return 1;
        }
        else {
            float deltaS = other.score - this.score;
            if (deltaS < 0.0F) {
                return -1;
            } else if (deltaS > 0.0F){
                return 1;
            }
            else {
                float deltaV = other.nbVisits - this.nbVisits;
                if (deltaV < 0.0F) {
                    return -1;
                } else {
                    return deltaV > 0.0F ? 1 : 0;
                }
            }
        }
    }

    /**
     * Compare function to see which CompletedMove performs better as described by Cohen-Solal. But this time, the
     * move with the least amount of visits is preferred. The order is based on Completion, value, and the
     * negative number of visits
     *
     * @param other the CompletedMove to be compared.
     * @return
     */
    public int compareToNegVis(CompletedMove other) {
        float deltaC = other.completion - this.completion;
        if (deltaC < 0.0F) {
            return -1;
        } else if (deltaC > 0.0F){
            return 1;
        }
        else {
            float deltaS = other.score - this.score;
            if (deltaS < 0.0F) {
                return -1;
            } else if (deltaS > 0.0F){
                return 1;
            }
            else {
                float deltaV = other.nbVisits - this.nbVisits;
                if (deltaV < 0.0F) {
                    return 1;
                } else {
                    return deltaV > 0.0F ? -1 : 0;
                }
            }
        }
    }

    /**
     * Repositions the given CompletedMove to the correct position in a sorted list of completed moves based on the
     * new score.
     *
     * @param move The move that has been made
     * @param resolution The resolution (is the game position solved or not)
     * @param completion The completion (is the game a loss, draw or win? (-1, 0, 1, respectively))
     * @param outputScore The new score of the given move (based on the GameStateEvaluator)
     * @param nbV The old number of times this moves has been visited
     * @param sortedCompletedMoves Sorted list of completed moves
     * @param bestIndex The old index of the completedMove
     * @param numLegalMoves Number of legal moves
     * @param mover The player to move
     * @param maximisingPlayer The maximising player.
     * @return Correctly sorted list of completed moves including the changed value
     */
    public static List<CompletedMove> addScoreToSortedCompletedMoves(Move move, float resolution, float completion,
                                                               float outputScore, int nbV,
                                                               List<CompletedMove> sortedCompletedMoves,
                                                               int bestIndex, int numLegalMoves, int mover,
                                                               int maximisingPlayer) {
        // Finding correct spot in sortedScored moves to place the new move (start by copying from current spot
        // of move).
        CompletedMove newCompletedMove = new CompletedMove(move, resolution, completion, outputScore, nbV+1);
        int k = bestIndex;
        label99:
        while (true) {
            while (true) {
                // If out of index stop
                if (k >= numLegalMoves - 1) {
                    break label99;
                }

                // -1 newCompleted better, 0 equal, 1 other is better
                // Check if next is better than new (shift new to the back)
                if ((newCompletedMove.compareTo(sortedCompletedMoves.get(k+1)) == 1 && mover == maximisingPlayer) ||
                        (newCompletedMove.compareToNegVis(sortedCompletedMoves.get(k+1)) == -1 && mover != maximisingPlayer)){
                    sortedCompletedMoves.set(k, sortedCompletedMoves.get(k + 1));
                    ++k;
                } else {
                    // Check if previous is better than new, if so, stop, else continue
                    if (k <= 0 ||
                            (newCompletedMove.compareTo(sortedCompletedMoves.get(k-1)) == 1 && mover == maximisingPlayer) ||
                            (newCompletedMove.compareToNegVis(sortedCompletedMoves.get(k-1)) == -1 && mover != maximisingPlayer)){
                        break label99;
                    }

                    sortedCompletedMoves.set(k, sortedCompletedMoves.get(k - 1));
                    --k;

                }
            }
        }

        // Set new value at correct position
        sortedCompletedMoves.set(k, new CompletedMove(move, resolution, completion, outputScore, nbV + 1));

        // Return re-ordered sortedScoredMoves
        return sortedCompletedMoves;
    }

    /**
     * Back-up the resolution. Checks if the node is completed or if all children are resolved. Proposed in:
     * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
     * knowledge. arXiv preprint arXiv:2008.01188.
     *
     * @param completion New completion value of current node
     * @param sortedCompletedMoves Sorted list of completed moves
     * @return Correct new resolution value for the current node
     */
    public static float backupResolution(float completion, List<CompletedMove> sortedCompletedMoves){
        // If completed, it is resolved
        if (completion != 0){
            return 1;
        }
        // Else, all children need to be resolved. Otherwise, the node is not resolved
        else {
            for (int i = 0; i < sortedCompletedMoves.size(); i++) {
                if (sortedCompletedMoves.get(i).resolution == 0){
                    return 0;
                }
            }
            return 1;
        }
    }

    /**
     * Get best completed action (used during the expansion of the tree). It selects the best CompletedMove
     * based on the best completion and score value, with the lowest number of iterations.
     * Two different selection strategies exist, based on the explorationPolicy selected.
     *
     * @param completedMoves Sorted list of completed moves
     * @param explorationPolicy Exploration policy (selection during search)
     * @param selectionEpsilon Epsilon of epsilon-greedy
     * @return Index of the best move according to the exploration strategy used
     */
    public static int getCompletedBestActionDual(List<CompletedMove> completedMoves,
                                                 ExplorationPolicy explorationPolicy,
                                                 float selectionEpsilon) {
        int indexPicked;
        switch (explorationPolicy) {
            case BEST:
                // Extract best value of unresolved moves
                float bestScore = 9999999;
                float bestCompletion = 9999999;
                int startIndex = 0;
                for (int i = 0; i < completedMoves.size(); i++) {
                    CompletedMove tempCompletedMove = completedMoves.get(i);
                    if (tempCompletedMove.resolution == 0){
                        bestCompletion = tempCompletedMove.completion;
                        bestScore = tempCompletedMove.score;
                        startIndex = i;
                        break;
                    }
                }

                // Check which move with equal value and completion has the fewest number of visits
                int index;
                for (index = startIndex+1; index < completedMoves.size(); index++) {
                    CompletedMove tempCompletedMove = completedMoves.get(index);
                    if (!(tempCompletedMove.completion == bestCompletion && tempCompletedMove.score == bestScore)) {
                        break;
                    }
                }

                // Get last with same values.
                indexPicked = index - 1;
                break;
            case EPSILON_GREEDY:
                // Take random with same completion value
                if (ThreadLocalRandom.current().nextDouble(1.0f) < selectionEpsilon) {
                    // Extract best value of unresolved moves
                    bestCompletion = 9999999;
                    startIndex = 0;
                    for (int i = 0; i < completedMoves.size(); i++) {
                        CompletedMove tempCompletedMove = completedMoves.get(i);
                        if (tempCompletedMove.resolution == 0){
                            bestCompletion = tempCompletedMove.completion;
                            startIndex = i;
                            break;
                        }
                    }

                    // Check which move with equal value and completion has the fewest number of visits
                    for (index = startIndex+1; index < completedMoves.size(); index++) {
                        CompletedMove tempCompletedMove = completedMoves.get(index);
                        if (!(tempCompletedMove.completion == bestCompletion)) {
                            break;
                        }
                    }

                    indexPicked = ThreadLocalRandom.current().nextInt(startIndex, index);
                } else {
                    // Extract best value of unresolved moves
                    bestScore = 9999999;
                    bestCompletion = 9999999;
                    startIndex = 0;
                    for (int i = 0; i < completedMoves.size(); i++) {
                        CompletedMove tempCompletedMove = completedMoves.get(i);
                        if (tempCompletedMove.resolution == 0) {
                            bestCompletion = tempCompletedMove.completion;
                            bestScore = tempCompletedMove.score;
                            startIndex = i;
                            break;
                        }
                    }

                    // Check which move with equal value and completion has the fewest number of visits
                    for (index = startIndex+1; index < completedMoves.size(); index++) {
                        CompletedMove tempCompletedMove = completedMoves.get(index);
                        if (!(tempCompletedMove.completion == bestCompletion && tempCompletedMove.score == bestScore)) {
                            break;
                        }
                    }

                    // Get last with same values.
                    indexPicked = index - 1;
                }
                break;
            default:
                throw new RuntimeException("Unkown exploration policy");
        }

        return indexPicked;
    }

    /**
     * Get best Completed Action (used to save correct values to TT). Since the list of completed moves is always
     * sorted, no calculation need to take place, making the iterations really fast!
     *
     * @return Index of the best move (the first one)
     */
    public static int getCompletedBestAction() {
        return 0;
    }
}
