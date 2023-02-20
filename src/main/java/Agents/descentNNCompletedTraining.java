package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import game.Game;
import main.collections.FastArrayList;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
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

/** Selects the best move to play based by using batched Neural Network evaluations in
 * combination with completed descent as proposed in Cohen-Solal, Q. (2020). Learning to play two-player perfect-information
 * games without knowledge. arXiv preprint arXiv:2008.01188. It is implemented to additionally store the trainings data
 * found after searching. This allows the search algorithm to be used in the descent framework */
public class descentNNCompletedTraining extends NNBot {

    //-------------------------------------------------------------------------

    /** Player ID indicating which player this bot is (1 for player 1, 2 for player 2, etc.) */
    protected int player = -1;

    /** Indicates the exploration policy (selection during search) used */
    protected Enums.ExplorationPolicy explorationPolicy;

    /** Indicates the epsilon of epsilon-greedy (when used) */
    protected final float explorationEpsilon = .05f;

    /** Transposition Table used to store the nodes */
    protected TranspositionTableStampCompleted TT = null;

    /** Number of bits used for primary key of the transposition table */
    protected int numBitsPrimaryCode = 12;

    /** Number of iterations performed by the bot during the last search */
    protected int iterations;

    /** Indicates the selection policy (selection of final move) used */
    protected Enums.SelectionPolicy selectionPolicy;

    /** GameStateEvaluator used to evaluate non-terminal leaf nodes, should be a neural network */
    protected final ClassicTerminalStateEvaluator classicTerminalStateEvaluator = new ClassicTerminalStateEvaluator();

    //-------------------------------------------------------------------------

    /**
     * Constructor with MultiLayerNetwork (DeepLearning4J) as input (uses epsilon-greedy exploration policy and
     * safest selection policy).
     * @param net MultiLayerNetwork (DeepLearning4J) used for training
     */
    public descentNNCompletedTraining(MultiLayerNetwork net) {
        this.friendlyName = "descent Completed (Neural Network)";
        this.explorationPolicy = Enums.ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = Enums.SelectionPolicy.SAFEST;

        this.net = net;
    }

    public descentNNCompletedTraining() {
    }

    /**
     * Selects and returns an action to play based on completed descent. The search algorithm evaluates all children
     * batched (which increases the iterations when using NNs (compared to individual evaluations)).
     *
     * @param game Reference to the game we're playing.
     * @param context Copy of the context containing the current state of the game
     * @param MaxSeconds Max number of seconds before a move should be selected.
     * Values less than 0 mean there is no time limit.
     * @param maxIterations Max number of iterations before a move should be selected.
     * Values less than 0 mean there is no iteration limit.
     * @param maxDepth Max search depth before a move should be selected.
     * Values less than 0 mean there is no search depth limit.
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
            descent_iteration(contextNew, maximisingPlayer, stopTime, 0);
            iterations++;
        }

        // Print iterations (uncomment if wished)
//        System.out.println("+ " + iterations);

        // Load rootTableData
        StampTTDataCompleted rootTableData = this.TT.retrieve(context.state().fullHash(context));

        // Remove old stamps and update stamp
        this.TT.deallocateOldStamps();
        this.TT.updateStamp();

        // Return the move according to the selection strategy.
        return this.finalMoveSelection(rootTableData,
                context.state().playerToAgent(context.state().mover()) == maximisingPlayer).move;
    }

    /**
     * Performs single iteration of the completed descent algorithm. The search algorithm evaluates all children
     * batched (which increases the iterations when using NNs (compared to individual evaluations)), while also
     * keeping track of the trainings data in the designated transposition table.
     *
     * @param context Copy of the context containing the current state of the game
     * @param maximisingPlayer ID of the player to maximise (always player one)
     * @param stopTime The time to terminate the iteration
     * @param depth Current depth of UBFM
     * @return Backpropagated estimated value, indicating how good the position is
     */
    private float descent_iteration(Context context, final int maximisingPlayer, final long stopTime, int depth) {
        float outputScore;
        INDArray inputNN = null;
        long zobrist = context.state().fullHash(context);
        if (context.trial().over()) {
            // Determine score
            outputScore = this.terminalEvaluator.evaluate(context, 1);

            // Add state to Transposition table
            this.TT.store(zobrist, 1, this.classicTerminalStateEvaluator.evaluate(context, 1),
                    outputScore, depth - 1, null);

            if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                inputNN = this.leafEvaluator.boardToInput(context);
                this.TTTraining.store(zobrist, outputScore, depth, inputNN.data().asFloat());
            }
        } else {
            // Check if state is in Transposition Table
            List<CompletedMove> sortedCompletedMoves = null;
            StampTTDataCompleted tableData = this.TT.retrieve(zobrist);

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

                        if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                            inputNN = this.leafEvaluator.boardToInput(contextCopy);
                            this.TTTraining.store(zobristCopy, moveScore, depth, inputNN.data().asFloat());
                        }
                    } else {
                        nonTerminalMoves.add(i);
                    }
                }

                // Calculate as batch
                if (nonTerminalMoves.size() > 0) {
                    float[] nonTerminalMoveScores = this.leafEvaluator.evaluateMoves(context,
                            nonTerminalMoves, maximisingPlayer);
                    for (int i = 0; i < nonTerminalMoves.size(); i++) {
                        tempScoredMoves.add(new CompletedMove(legalMoves.get(nonTerminalMoves.get(i)), 0,
                                0, nonTerminalMoveScores[i], 1));
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

                // Update completion and resolution value and save to TT
                int bestIndex = getCompletedBestAction();
                CompletedMove bestCompletedMove = sortedCompletedMoves.get(bestIndex);
                float resolution = backupResolution(bestCompletedMove.completion, sortedCompletedMoves);
                outputScore = bestCompletedMove.score;

                this.TT.store(zobrist, resolution, bestCompletedMove.completion, outputScore,
                        depth - 1, sortedCompletedMoves);

                if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                    inputNN = this.leafEvaluator.boardToInput(context);
                    this.TTTraining.store(zobrist, outputScore, depth, inputNN.data().asFloat());
                }

                // Update tableData to new data
                tableData = this.TT.retrieve(zobrist);
            }

            if (tableData.resolution == 0) {
                // Get an action to play
                int bestIndex = getCompletedBestActionDual(sortedCompletedMoves, this.explorationPolicy,
                        this.explorationEpsilon);
                Move bestMove = sortedCompletedMoves.get(bestIndex).move;
                int nbVisits = sortedCompletedMoves.get(bestIndex).nbVisits;

                // Perform move and perform a new iteration
                Context contextCopy = new Context(context);
                contextCopy.game().apply(contextCopy, bestMove);
                long bestZobrist = contextCopy.state().fullHash(contextCopy);
                outputScore = this.descent_iteration(contextCopy, maximisingPlayer, stopTime, depth + 1);

                // Update score and visits (also considering the new completion and resolution
                StampTTDataCompleted bestData = this.TT.retrieve(bestZobrist);
                sortedCompletedMoves = addScoreToSortedCompletedMoves(bestMove, bestData.resolution,
                        bestData.completion, outputScore, nbVisits, sortedCompletedMoves, bestIndex,
                        numLegalMoves, context.state().playerToAgent(context.state().mover()), maximisingPlayer);

                // Get best action to update completion, score and resolution
                bestIndex = getCompletedBestAction();
                CompletedMove bestCompletedMove = sortedCompletedMoves.get(bestIndex);
                float resolution = backupResolution(bestCompletedMove.completion, sortedCompletedMoves);
                outputScore = bestCompletedMove.score;

                // Save all changes to TT
                this.TT.store(zobrist, resolution, bestCompletedMove.completion, outputScore,
                        depth - 1, sortedCompletedMoves);

                if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                    inputNN = this.leafEvaluator.boardToInput(context);
                    this.TTTraining.store(zobrist, outputScore, depth, inputNN.data().asFloat());
                }
            }
            else {
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
     * @param game The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;
        this.TT = new TranspositionTableStampCompleted(numBitsPrimaryCode);
        this.TT.allocate();

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, this.net);
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
    }

    /**
     * Get the best action to play in the actual game based on the selection policy used
     *
     * @param rootTableData Table from the root node in the transposition table
     * @param maximising Indicates if the player is maximising
     * @return The best scored move according to the selection policy
     */
    protected CompletedMove finalMoveSelection(StampTTDataCompleted rootTableData, boolean maximising) {
        List<CompletedMove> sortedCompletedMoves = rootTableData.sortedScoredMoves;

        switch (this.selectionPolicy) {
            case BEST:
                if (maximising){
                    return sortedCompletedMoves.stream().max(Comparator.comparing(CompletedMove::getCompletion)
                            .thenComparing(CompletedMove::getScore)
                            .thenComparing(CompletedMove::getNbVisits)).get();
                } else {
                    return sortedCompletedMoves.stream().min(Comparator.comparing(CompletedMove::getCompletion)
                            .thenComparing(CompletedMove::getScore)
                            .thenComparing(CompletedMove::getNegativeNbVisits)).get();
                }
            case SAFEST:
                if (maximising){
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
}


