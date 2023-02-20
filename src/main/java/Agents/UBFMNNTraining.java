package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import game.Game;
import main.collections.FVector;
import main.collections.FastArrayList;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import other.context.Context;
import other.move.Move;
import utils.Enums;
import utils.TranspositionTableStamp;
import utils.data_structures.ScoredMove;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static utils.descentUtils.*;

/** Selects the best move to play based by using batched Neural Network evaluations in
 * combination with UBFM as proposed in Cohen-Solal, Q. (2020). Learning to play two-player perfect-information
 * games without knowledge. arXiv preprint arXiv:2008.01188. It is implemented to additionally store the trainings data
 * found after searching. This allows the search algorithm to be used in the descent framework */
public class UBFMNNTraining extends NNBot {

    //-------------------------------------------------------------------------

    /** Player ID indicating which player this bot is (1 for player 1, 2 for player 2, etc.) */
    protected int player = -1;;

    /** Indicates the exploration policy (selection during search) used */
    protected Enums.ExplorationPolicy explorationPolicy;

    /** Indicates the epsilon of epsilon-greedy (when used) */
    protected final float explorationEpsilon = .05f;

    /** Transposition Table used to store the nodes */
    protected TranspositionTableStamp TT = null;

    /** Number of bits used for primary key of the transposition table */
    protected int numBitsPrimaryCode = 12;

    /** Number of iterations performed by the bot during the last search */
    protected int iterations;

    /** Indicates the selection policy (selection of final move) used */
    protected Enums.SelectionPolicy selectionPolicy;

    //-------------------------------------------------------------------------

    /**
     * Constructor with MultiLayerNetwork (DeepLearning4J) as input (uses epsilon-greedy exploration policy and
     * safest selection policy).
     * @param net MultiLayerNetwork (DeepLearning4J) used for training
     */
    public UBFMNNTraining(MultiLayerNetwork net) {
        this.friendlyName = "UBFM (Neural Network)";
        this.explorationPolicy = Enums.ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = Enums.SelectionPolicy.BEST;

        this.net = net;
    }

    /**
     * Selects and returns an action to play based on UBFM. The search algorithm evaluates all children
     * batched (which increases the iterations when using NNs (compared to individual evaluations))
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
            UBFM_iteration(contextNew, maximisingPlayer, stopTime, 0);
            iterations++;
        }

        // Load rootTableData
        TranspositionTableStamp.StampTTData rootTableData = this.TT.retrieve(context.state().fullHash(context));

        // Remove old stamps and update stamp
        this.TT.deallocateOldStamps();
        this.TT.updateStamp();

        // Return the move according to the selection strategy.
        return finalMoveSelection(rootTableData, this.selectionPolicy,
                context.state().playerToAgent(context.state().mover()) == maximisingPlayer).move;
    }

    /**
     * Performs single iteration of the UBFM algorithm. The search algorithm evaluates all children
     * batched (which speeds up the iterations / sec when using NNs), while also keeping track of the traings data
     * in the designated transposition table.
     *
     * @param context Copy of the context containing the current state of the game
     * @param maximisingPlayer ID of the player to maximise (always player one)
     * @param stopTime The time to terminate the iteration
     * @param depth Current depth of UBFM
     * @return Backpropagated estimated value, indicating how good the position is
     */
    private float UBFM_iteration(Context context, final int maximisingPlayer, final long stopTime, int depth) {
        float outputScore;
        INDArray inputNN = null;
        long zobrist = context.state().fullHash(context);
        if (context.trial().over()) {
            // Determine score
            outputScore = this.terminalEvaluator.evaluate(context, 1);

            // Add state to Transposition table
            this.TT.store(zobrist, outputScore, depth - 1, null);

            if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                inputNN = this.leafEvaluator.boardToInput(context);
                this.TTTraining.store(zobrist, outputScore, depth, inputNN.data().asFloat());
            }
        } else {
            // Check if state is in Transposition Table
            List<ScoredMove> sortedScoredMoves = null;
            TranspositionTableStamp.StampTTData tableData = this.TT.retrieve(zobrist);
            if (tableData != null && tableData.sortedScoredMoves != null) {
                sortedScoredMoves = new ArrayList(tableData.sortedScoredMoves);
            }

            // Get all legal moves
            FastArrayList<Move> legalMoves = context.moves(context).moves();
            int numLegalMoves = legalMoves.size();

            // If nothing has been found
            if (sortedScoredMoves == null) {
                // Loop over all legal moves
                float moveScore;
                FVector moveScores = new FVector(numLegalMoves);
                int mover = context.state().playerToAgent(context.state().mover());
                ArrayList<Integer> nonTerminalMoves = new ArrayList<Integer>();
                for (int i = 0; i < numLegalMoves; i++) {
                    Context contextCopy = new Context(context);
                    contextCopy.game().apply(contextCopy, legalMoves.get(i));

                    if (contextCopy.trial().over()) {
                        // Determine terminalEvaluation
                        moveScore = this.terminalEvaluator.evaluate(contextCopy, 1); // terminalEvaluation

                        // Add to TT (for terminal state)
                        long zobristCopy = contextCopy.state().fullHash(contextCopy);
                        this.TT.store(zobristCopy, moveScore, depth - 1, null);

                        if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                            inputNN = this.leafEvaluator.boardToInput(contextCopy);
                            this.TTTraining.store(zobristCopy, moveScore, depth, inputNN.data().asFloat());
                        }

                        moveScores.set(i, moveScore);
                    } else {
                        nonTerminalMoves.add(i);
                    }
                }

                // If time left, calculate as batch
                if (System.currentTimeMillis() >= stopTime || this.wantsInterrupt && nonTerminalMoves.size() > 0) {
                    for (int i = 0; i < nonTerminalMoves.size(); i++) {
                        moveScores.set(nonTerminalMoves.get(i), mover == maximisingPlayer ? -999999.0F : 999999.0F);
                    }
                } else if (nonTerminalMoves.size() > 0) {
                    float[] nonTerminalMoveScores = this.leafEvaluator.evaluateMoves(context,
                            nonTerminalMoves, maximisingPlayer);
                    for (int i = 0; i < nonTerminalMoves.size(); i++) {
                        moveScores.set(nonTerminalMoves.get(i), nonTerminalMoveScores[i]);
                    }
                }

                // Finally, sort all scores and save as ScoredMove
                FastArrayList<ScoredMove> tempScoredMoves = new FastArrayList(numLegalMoves);
                for (int i = 0; i < numLegalMoves; i++) {
                    tempScoredMoves.add(new ScoredMove(legalMoves.get(i), moveScores.get(i), 1));
                }

                sortedScoredMoves = new ArrayList(numLegalMoves);

                for (int i = 0; i < numLegalMoves; ++i) {
                    sortedScoredMoves.add((ScoredMove) tempScoredMoves.removeSwap(ThreadLocalRandom.current().nextInt(tempScoredMoves.size())));
                }

                if (mover == maximisingPlayer) {
                    Collections.sort(sortedScoredMoves);
                } else {
                    Collections.sort(sortedScoredMoves, Collections.reverseOrder());
                }

                // Save to TT
                this.TT.store(zobrist, sortedScoredMoves.get(0).score, depth - 1, sortedScoredMoves);

                if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                    inputNN = this.leafEvaluator.boardToInput(context);
                    this.TTTraining.store(zobrist, sortedScoredMoves.get(0).score, depth, inputNN.data().asFloat());
                }
            } else {
                // Get best value and move (save to TT)
                int bestIndex = getBestAction(sortedScoredMoves, numLegalMoves,
                        this.explorationPolicy, this.explorationEpsilon);
                Move bestMove = sortedScoredMoves.get(bestIndex).move;
                int nbVisits = sortedScoredMoves.get(bestIndex).nbVisits;

                // Perform move and perform a new iteration
                Context contextCopy = new Context(context);
                contextCopy.game().apply(contextCopy, bestMove);
                outputScore = this.UBFM_iteration(contextCopy, maximisingPlayer, stopTime, depth + 1);

                // Reposition scoredMove of calculated outputScore in sortedScoredMoves
                sortedScoredMoves = addScoreToSortedScoredMoves(bestMove, outputScore, nbVisits, sortedScoredMoves, bestIndex,
                        numLegalMoves, context.state().playerToAgent(context.state().mover()), maximisingPlayer);
            }

            // Get best action (No epsilon-greedy)
            outputScore = sortedScoredMoves.get(0).score;

            // Save to TT
            this.TT.store(zobrist, outputScore, depth - 1, sortedScoredMoves);
            if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                inputNN = this.leafEvaluator.boardToInput(context);
                this.TTTraining.store(zobrist, outputScore, depth, inputNN.data().asFloat());
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
        this.TT = new TranspositionTableStamp(numBitsPrimaryCode);
        this.TT.allocate();

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, this.net);
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
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


