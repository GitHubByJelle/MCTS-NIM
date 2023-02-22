package Agents.TestAgent;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.GameStateEvaluator;
import Evaluator.HeuristicLeafEvaluator;
import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;
import utils.CompletedMove;
import utils.Enums;
import utils.TranspositionTableStampCompleted;
import utils.TranspositionTableStampCompleted.StampTTDataCompleted;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static utils.CompletedMove.*;

/**
 * Selects the best move to play based by using Ludii's heuristic evaluation function in
 * combination with completed UBFM as proposed in Cohen-Solal, Q. (2020). Learning to play two-player perfect-information
 * games without knowledge. arXiv preprint arXiv:2008.01188. The "completion" adds a solver to the search. Please note,
 * the evaluations aren't batched, which can result in low performance when using a neural network.
 */
public class UBFMHFCompleted extends AI {

    //-------------------------------------------------------------------------

    /**
     * Player ID indicating which player this bot is (1 for player 1, 2 for player 2, etc.)
     */
    protected int player = -1;

    /**
     * Indicates the exploration policy (selection during search) used
     */
    protected Enums.ExplorationPolicy explorationPolicy;

    /**
     * Indicates the epsilon of epsilon-greedy (when used)
     */
    protected final float explorationEpsilon = .05f;

    /**
     * Transposition Table used to store the nodes with completion
     */
    protected TranspositionTableStampCompleted TT = null;

    /**
     * Number of bits used for primary key of the transposition table
     */
    protected int numBitsPrimaryCode = 12;

    /**
     * Number of iterations performed by the bot during the last search
     */
    protected int iterations;

    /**
     * Indicates the selection policy (selection of final move) used
     */
    protected Enums.SelectionPolicy selectionPolicy;

    /**
     * GameStateEvaluator used to evaluate non-terminal leaf nodes
     */
    protected GameStateEvaluator leafEvaluator;

    /**
     * GameStateEvaluator used to evaluate terminal leaf nodes
     */
    protected GameStateEvaluator terminalEvaluator;

    /**
     * GameStateEvaluator used to evaluate terminal leaf nodes with 1, 0 or -1
     */
    protected final ClassicTerminalStateEvaluator classicTerminalStateEvaluator = new ClassicTerminalStateEvaluator();

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (uses epsilon-greedy exploration policy and safest selection policy).
     */
    public UBFMHFCompleted() {
        this.friendlyName = "UBFM Completed (Heuristic)";
        this.explorationPolicy = Enums.ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = Enums.SelectionPolicy.SAFEST;
    }

    /**
     * Selects and returns an action to play based on UBFM with completion. The search algorithm evaluates all children
     * individually (which could be a disadvantage when using NNs).
     *
     * @param game          Reference to the game we're playing.
     * @param context       Copy of the context containing the current state of the game
     * @param MaxSeconds    Max number of seconds before a move should be selected.
     *                      Values less than 0 mean there is no time limit.
     * @param maxIterations Max number of iterations before a move should be selected.
     *                      Values less than 0 mean there is no iteration limit.
     * @param maxDepth      Max search depth before a move should be selected.
     *                      Values less than 0 mean there is no search depth limit.
     * @return Preferred move.
     */
    @Override
    public Move selectAction
    (
            final Game game, final Context context, final double MaxSeconds,
            final int maxIterations, final int maxDepth
    ) {
        // Determine maximum iterations and stop time
        long stopTime = (MaxSeconds > 0.0) ? System.currentTimeMillis() + (long) (MaxSeconds * 1000) : Long.MAX_VALUE;
        int maxIts = (maxIterations >= 0) ? maxIterations : Integer.MAX_VALUE;

        // Extract the player to maximise (always player 1)
        final int maximisingPlayer = context.state().playerToAgent(1);
        Context contextNew = new Context(context);

        // Reset iterations
        iterations = 0;

        // Perform UBFM iterations until no time is left
        while (System.currentTimeMillis() < stopTime && iterations < maxIts && !wantsInterrupt) {
            UBFM_iteration(contextNew, maximisingPlayer, stopTime, 0);
            iterations++;
        }

        // Print iterations (uncomment if wished)
//        System.out.println(iterations);

        // Load rootTableData to use during final move selection
        StampTTDataCompleted rootTableData = this.TT.retrieve(context.state().fullHash(context));

        // Remove old stamps and update stamp
        this.TT.deallocateOldStamps();
        this.TT.updateStamp();

        // Return the move according to the selection strategy
        return this.finalMoveSelection(rootTableData,
                context.state().playerToAgent(context.state().mover()) == maximisingPlayer).move;
    }

    /**
     * Performs single iteration of the completed UBFM algorithm. The search algorithm evaluates all children
     * individually (which could be a disadvantage when using NNs).
     *
     * @param context          Copy of the context containing the current state of the game
     * @param maximisingPlayer ID of the player to maximise (always player one)
     * @param stopTime         The time to terminate the iteration
     * @param depth            Current depth of UBFM
     * @return Backpropagated estimated value, indicating how good the position is
     */
    protected float UBFM_iteration(Context context, final int maximisingPlayer, final long stopTime, int depth) {
        float outputScore;
        long zobrist = context.state().fullHash(context);
        if (context.trial().over()) {
            // Determine score
            outputScore = this.terminalEvaluator.evaluate(context, 1);

            // Add state to Transposition table
            this.TT.store(zobrist, 1, this.classicTerminalStateEvaluator.evaluate(context, 1),
                    outputScore, depth - 1, null);
        } else {
            // Check if state is in Transposition Table
            List<CompletedMove> sortedCompletedMoves = null;
            StampTTDataCompleted tableData = this.TT.retrieve(zobrist);
            if (tableData == null || tableData.resolution == 0) {
                if (tableData != null && tableData.sortedScoredMoves != null) {
                    sortedCompletedMoves = new ArrayList(tableData.sortedScoredMoves);
                }

                // Get all legal moves
                FastArrayList<Move> legalMoves = context.moves(context).moves();
                int numLegalMoves = legalMoves.size();

                // If nothing has been found
                if (sortedCompletedMoves == null) {
                    // Loop over all legal moves
                    float moveScore;
                    int mover = context.state().playerToAgent(context.state().mover());
                    FastArrayList<CompletedMove> tempScoredMoves = new FastArrayList(numLegalMoves);
                    ArrayList<Integer> nonTerminalMoves = new ArrayList<Integer>();
                    for (int i = 0; i < numLegalMoves; i++) {
                        Context contextCopy = new Context(context);
                        contextCopy.game().apply(contextCopy, legalMoves.get(i));

                        if (contextCopy.trial().over()) {
                            // Determine terminalEvaluation
                            moveScore = this.terminalEvaluator.evaluate(contextCopy, 1); // terminalEvaluation

                            // Add to TT (for terminal state)
                            long zobristCopy = contextCopy.state().fullHash(contextCopy);
                            this.TT.store(zobristCopy, 1,
                                    this.classicTerminalStateEvaluator.evaluate(contextCopy, 1),
                                    moveScore, depth - 1, null);

                            tempScoredMoves.add(new CompletedMove(legalMoves.get(i), 1,
                                    this.classicTerminalStateEvaluator.evaluate(contextCopy, 1),
                                    moveScore, 1));
                        } else {
                            tempScoredMoves.add(new CompletedMove(legalMoves.get(i), 0, 0,
                                    this.leafEvaluator.evaluate(contextCopy, maximisingPlayer), 1));
                        }
                    }

                    // Finally, sort all scores and save as ScoredMove
                    sortedCompletedMoves = new ArrayList(numLegalMoves);
                    for (int i = 0; i < numLegalMoves; ++i) {
                        sortedCompletedMoves.add((CompletedMove) tempScoredMoves.removeSwap(ThreadLocalRandom.current().nextInt(tempScoredMoves.size())));
                    }

                    if (mover == maximisingPlayer) {
                        Collections.sort(sortedCompletedMoves);
                    } else {
                        Collections.sort(sortedCompletedMoves, Collections.reverseOrder());
                    }

                    // Save to TT
                    CompletedMove completedMove = sortedCompletedMoves.get(0);
                    this.TT.store(zobrist, completedMove.resolution, completedMove.completion,
                            completedMove.score, depth - 1, sortedCompletedMoves);
                } else {
                    // Get best value and move (save to TT)
                    int bestIndex = getCompletedBestActionDual(sortedCompletedMoves, this.explorationPolicy,
                            this.explorationEpsilon);
                    Move bestMove = sortedCompletedMoves.get(bestIndex).move;
                    int nbVisits = sortedCompletedMoves.get(bestIndex).nbVisits;

                    // Perform move and perform a new iteration
                    Context contextCopy = new Context(context);
                    contextCopy.game().apply(contextCopy, bestMove);
                    long bestZobrist = contextCopy.state().fullHash(contextCopy);
                    outputScore = this.UBFM_iteration(contextCopy, maximisingPlayer, stopTime, depth + 1);

                    // Update score and visits (also considering the new completion and resolution
                    StampTTDataCompleted bestData = this.TT.retrieve(bestZobrist);
                    sortedCompletedMoves = addScoreToSortedCompletedMoves(bestMove, bestData.resolution,
                            bestData.completion, outputScore, nbVisits, sortedCompletedMoves, bestIndex,
                            numLegalMoves, context.state().playerToAgent(context.state().mover()), maximisingPlayer);
                }

                // Update completion and resolution value
                int bestIndex = getCompletedBestAction();
                CompletedMove bestCompletedMove = sortedCompletedMoves.get(bestIndex);
                float resolution = backupResolution(bestCompletedMove.completion, sortedCompletedMoves);

                outputScore = bestCompletedMove.score;

                this.TT.store(zobrist, resolution, bestCompletedMove.completion, outputScore,
                        depth - 1, sortedCompletedMoves);
            } else {
                outputScore = tableData.value;
            }
        }

        // Return outputScore
        return outputScore;
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Set the playerID, initialise a new Transposition Table and initialise both GameStateEvaluators
     *
     * @param game     The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;
        this.TT = new TranspositionTableStampCompleted(numBitsPrimaryCode);
        this.TT.allocate();

        this.leafEvaluator = new HeuristicLeafEvaluator(game);
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
    }

    /**
     * Get the best action to play in the actual game
     *
     * @param rootTableData Table from the root node in the transposition table
     * @param maximising    Indicates if the player is maximising
     * @return The best scored move according to the selection policy
     */
    protected CompletedMove finalMoveSelection(StampTTDataCompleted rootTableData, boolean maximising) {
        List<CompletedMove> sortedCompletedMoves = rootTableData.sortedScoredMoves;

        switch (this.selectionPolicy) {
            case BEST:
                if (maximising) {
                    return sortedCompletedMoves.stream().max(Comparator.comparing(CompletedMove::getCompletion)
                            .thenComparing(CompletedMove::getScore)
                            .thenComparing(CompletedMove::getNbVisits)).get();
                } else {
                    return sortedCompletedMoves.stream().min(Comparator.comparing(CompletedMove::getCompletion)
                            .thenComparing(CompletedMove::getScore)
                            .thenComparing(CompletedMove::getNegativeNbVisits)).get();
                }
            case SAFEST:
                if (maximising) {
                    return sortedCompletedMoves.stream().max(Comparator.comparing(CompletedMove::getCompletion)
                            .thenComparing(CompletedMove::getNbVisits)
                            .thenComparing(CompletedMove::getScore)).get();
                } else {
                    return sortedCompletedMoves.stream().min(Comparator.comparing(CompletedMove::getCompletion)
                            .thenComparing(CompletedMove::getNegativeNbVisits)
                            .thenComparing(CompletedMove::getScore)).get();
                }
            default:
                System.err.println("Error: selectionPolicy not implemented");
                return rootTableData.sortedScoredMoves.get(0);
        }
    }

    /**
     * Allows an agent to tell Ludii whether or not it can support playing
     * any given game. Copied from the UBFM implementation from Ludii.
     *
     * @param game Ludii's game
     * @return False if the AI cannot play the given game.
     */
    public boolean supportsGame(Game game) {
        if (game.isStochasticGame()) {
            return false;
        } else if (game.hiddenInformation()) {
            return false;
        } else if (game.hasSubgames()) {
            return false;
        } else {
            return game.isAlternatingMoveGame();
        }
    }
}


