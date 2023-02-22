package Agents;

import Agents.TestAgent.UBFMHF;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import Training.LearningManager;
import game.Game;
import main.collections.FVector;
import main.collections.FastArrayList;
import other.context.Context;
import other.move.Move;
import utils.Enums;
import utils.TranspositionTableStamp;
import utils.data_structures.ScoredMove;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static utils.descentUtils.addScoreToSortedScoredMoves;
import static utils.descentUtils.getBestAction;

/**
 * Selects the best move to play based by using batched Neural Network evaluations in
 * combination with UBFM as proposed in Cohen-Solal, Q. (2020). Learning to play two-player perfect-information
 * games without knowledge. arXiv preprint arXiv:2008.01188.
 */
public class UBFMNN extends UBFMHF {

    //-------------------------------------------------------------------------

    /**
     * Neural Network evaluator used to evaluate non-terminal leaf nodes batched
     */
    protected NeuralNetworkLeafEvaluator leafEvaluator;

    /**
     * Path to the neural network to used by default
     */
    protected String pathName = "NN_models/Network_bSize128_nEp1_nGa1563_2022-11-12-04-50-34.bin";

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (uses epsilon-greedy exploration policy and safest selection policy).
     */
    public UBFMNN() {
        this.friendlyName = "UBFM (Neural Network)";
        this.explorationPolicy = Enums.ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = Enums.SelectionPolicy.SAFEST;
    }

    /**
     * Constructor with the path to the desired neural network as path
     * (uses epsilon-greedy exploration policy and safest selection policy).
     */
    public UBFMNN(String pathName) {
        this.friendlyName = "UBFM (Neural Network)";
        this.explorationPolicy = Enums.ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = Enums.SelectionPolicy.SAFEST;

        this.pathName = pathName;
    }

    /**
     * Performs single iteration of the UBFM algorithm. The search algorithm evaluates all children
     * batched (which speeds up the iterations / sec when using NNs).
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
            this.TT.store(zobrist, outputScore, depth - 1, null);
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

                        moveScores.set(i, moveScore);
                    } else {
                        nonTerminalMoves.add(i);
//                        moveScore = this.leafEvaluator.evaluate(contextCopy, maximisingPlayer); // leafEvaluation
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
        this.TT = new TranspositionTableStamp(numBitsPrimaryCode);
        this.TT.allocate();

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, LearningManager.loadNetwork(pathName, false));
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
    }
}


