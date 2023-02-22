package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import Training.LearningManager;
import game.Game;
import main.collections.FastArrayList;
import other.context.Context;
import other.move.Move;
import utils.CompletedMove;
import utils.Enums;
import utils.TranspositionTableStampCompleted;
import utils.TranspositionTableStampCompleted.StampTTDataCompleted;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static utils.CompletedMove.*;

/**
 * Selects the best move to play based by using batched Neural Network evaluations in
 * combination with completed descent. The "completion" adds a solver to the search. The evaluation is performed batched,
 * resulting more iterations when using a neural network (compared to individual evaluations)
 */
public class descentNNCompleted extends descentHFCompleted {

    //-------------------------------------------------------------------------

    /**
     * GameStateEvaluator used to evaluate non-terminal leaf nodes
     */
    protected NeuralNetworkLeafEvaluator leafEvaluator;

    /**
     * Path to the neural network to used by default
     */
    protected String pathName = "NN_models/Network_bSize128_nEp1_nGa1563_2022-11-12-04-50-34.bin";

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (uses epsilon-greedy exploration policy, safest selection policy and the
     * default network).
     */
    public descentNNCompleted() {
        this.friendlyName = "descent Completed (Neural Network)";
        this.explorationPolicy = Enums.ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = Enums.SelectionPolicy.SAFEST;
    }

    /**
     * Constructor with the path to the desired neural network as path (uses epsilon-greedy exploration policy and
     * safest selection policy)
     *
     * @param pathName Path to the neural network to be used
     */
    public descentNNCompleted(String pathName) {
        this.friendlyName = "descent Completed (Neural Network))";
        this.explorationPolicy = Enums.ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = Enums.SelectionPolicy.SAFEST;

        this.pathName = pathName;
    }

    /**
     * Performs single iteration of the completed descent algorithm. The search algorithm evaluates all children
     * batched (which increases the iterations when using NNs (compared to individual evaluations))
     *
     * @param context          Copy of the context containing the current state of the game
     * @param maximisingPlayer ID of the player to maximise (always player one)
     * @param stopTime         The time to terminate the iteration
     * @param depth            Current depth of UBFM
     * @return Backpropagated estimated value, indicating how good the position is
     */
    protected float descent_iteration(Context context, final int maximisingPlayer, final long stopTime, int depth) {
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

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, LearningManager.loadNetwork(pathName, false));
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
    }
}


