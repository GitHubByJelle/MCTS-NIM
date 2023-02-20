//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import game.Game;
import main.collections.FVector;
import main.collections.FastArrayList;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import other.context.Context;
import other.move.Move;
import other.state.State;
import other.trial.Trial;
import training.expert_iteration.ExItExperience;
import utils.Enums;
import utils.TranspositionTableLearning;
import utils.data_structures.transposition_table.TranspositionTable;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Implementation of Ludii's alpha-beta search. Assumes perfect-information games.
 * Uses iterative deepening when time-restricted, goes straight for
 * depth limit when only depth-limited. Extracts heuristics to use from game's metadata.
 *
 * Adapted to work with all implemented GameStateEvaluators. Please note, the evaluations aren't batched, which can
 * result in low number of iterations when using a neural network. It is implemented
 * to additionally store the trainings data found after searching. This allows the search algorithm to be used in the
 * descent framework
 *
 * @author Dennis Soemers, adapted by Jelle Jansen
 */
public class AlphaBetaSearchNNTraining extends NNBot {

    //-------------------------------------------------------------------------

    /** We'll automatically return our move after at most this number of seconds if we only have one move */
    protected double autoPlaySeconds = 0.0;

    /** Estimated score of the root node based on last-run search */
    protected float estimatedRootScore = 0.f;

    /** The maximum heuristic eval we have ever observed */
    protected float maxHeuristicEval = 0.f;

    /** The minimum heuristic eval we have ever observed */
    protected float minHeuristicEval = 0.f;

    /** String to print to Analysis tab of the Ludii app */
    protected String analysisReport = null;

    /** Current list of moves available in root */
    protected FastArrayList<Move> currentRootMoves = null;

    /** The last move we returned. Need to memorise this for Expert Iteration with AlphaBeta */
    protected Move lastReturnedMove = null;

    /** Root context for which we've last performed a search */
    protected Context lastSearchedRootContext = null;

    /** Value estimates of moves available in root */
    protected FVector rootValueEstimates = null;

    /** The number of players in the game we're currently playing */
    protected int numPlayersInGame = 0;

    /** Remember if we proved a win in one of our searches */
    protected boolean provedWin = false;

    /** Needed for visualisations */
    protected float rootAlphaInit = -1.0F;

    /** Needed for visualisations */
    protected float rootBetaInit = 1.0F;

    /** Sorted (hopefully cleverly) list of moves available in root node */
    protected FastArrayList<Move> sortedRootMoves = null;

    /** If true at end of a search, it means we searched full tree (probably proved a draw) */
    protected boolean searchedFullTree = false;

    /** Do we want to allow using Transposition Table? */
    protected boolean allowTranspositionTable = true;

    /** Transposition Table */
    protected TranspositionTable transpositionTable = null;

    /** Do we allow any search depth, or only odd, or only even? */
    protected AllowedSearchDepths allowedSearchDepths;

    /** Number of iterations performed by the bot during the last search */
    private int iterations;

    //-------------------------------------------------------------------------

    /**
     * Constructor with MultiLayerNetwork (DeepLearning4J) as input
     */
    public AlphaBetaSearchNNTraining(MultiLayerNetwork net) {
        this.allowedSearchDepths = AllowedSearchDepths.Any;
        this.friendlyName = "Alpha-Beta with NN evaluation";

        this.net = net;
    }

    /**
     * Selects and returns an action to play based on Iterative Deepening. The search algorithm evaluates all children
     * individually (which could be a disadvantage when using NNs).
     *
     * @param game Reference to the game we're playing.
     * @param context Copy of the context containing the current state of the game
     * @param maxSeconds Max number of seconds before a move should be selected.
     * Values less than 0 mean there is no time limit.
     * @param maxIterations Max number of iterations before a move should be selected.
     * Values less than 0 mean there is no iteration limit.
     * @param maxDepth Max search depth before a move should be selected.
     * Values less than 0 mean there is no search depth limit.
     * @return Preferred move.
     */
    public Move selectAction(Game game, Context context, double maxSeconds, int maxIterations, int maxDepth) {
        this.provedWin = false;
        int depthLimit = maxDepth > 0 ? maxDepth : Integer.MAX_VALUE;
        this.lastSearchedRootContext = context;
        if (this.transpositionTable != null) {
            this.transpositionTable.allocate();
        }

        iterations = 0;
        int initDepth = this.allowedSearchDepths == AllowedSearchDepths.Even ? 2 : 1;
        this.lastReturnedMove = this.iterativeDeepening(game, context, maxSeconds, depthLimit, initDepth);

        if (this.transpositionTable != null) {
            this.transpositionTable.deallocate();
        }

        return this.lastReturnedMove;
    }

    /**
     * Runs iterative deepening alpha-beta
     *
     * @param game Reference to the game we're playing.
     * @param context Copy of the context containing the current state of the game
     * @param maxSeconds Max number of seconds before a move should be selected.
     * Values less than 0 mean there is no time limit.
     * @param maxDepth Max search depth before a move should be selected.
     * Values less than 0 mean there is no search depth limit.
     * @param startDepth The initial search depth
     * @return Preferred move.
     */
    public Move iterativeDeepening(Game game, Context context, double maxSeconds, int maxDepth, int startDepth) {
        long startTime = System.currentTimeMillis();
        long stopTime = maxSeconds > 0.0 ? startTime + (long) (maxSeconds * 1000.0) : Long.MAX_VALUE;
        this.currentRootMoves = new FastArrayList(game.moves(context).moves());
        FastArrayList<Move> tempMovesList = new FastArrayList(this.currentRootMoves);
        this.sortedRootMoves = new FastArrayList(this.currentRootMoves.size());

        // Create a shuffled version of list of moves (random tie-breaking)
        while (!tempMovesList.isEmpty()) {
            this.sortedRootMoves.add((Move) tempMovesList.removeSwap(ThreadLocalRandom.current().nextInt(tempMovesList.size())));
        }

        int numRootMoves = this.sortedRootMoves.size();
        List<ScoredMove> scoredMoves = new ArrayList(this.sortedRootMoves.size());

        // play faster if we only have one move available anyway
        if (numRootMoves == 1 && this.autoPlaySeconds >= 0.0 && this.autoPlaySeconds < maxSeconds) {
            stopTime = startTime + (long) (this.autoPlaySeconds * 1000.0);
        }

        // Vector for visualisation purposes
        this.rootValueEstimates = new FVector(this.currentRootMoves.size());

        // Storing scores found for purpose of move ordering
        FVector moveScores = new FVector(numRootMoves);

        int searchDepthIncrement = this.allowedSearchDepths == AllowedSearchDepths.Any ? 1 : 2;
        int searchDepth = startDepth - searchDepthIncrement;
        int maximisingPlayer = context.state().playerToAgent(context.state().mover());

        // Best move found so far during a fully-completed search
        // (ignoring incomplete early-terminated search)
        Move bestMoveCompleteSearch = (Move) this.sortedRootMoves.get(0);
        this.rootAlphaInit = -1.0F;
        this.rootBetaInit = 1.0F;

        while (searchDepth < maxDepth) {
            searchDepth += searchDepthIncrement;
            this.searchedFullTree = true;

            // the real alpha-beta stuff starts here
            float score = maximisingPlayer == 1 ? this.rootAlphaInit : this.rootBetaInit;
            float alpha = this.rootAlphaInit;
            float beta = this.rootBetaInit;

            // best move during this particular search
            Move bestMove = (Move) this.sortedRootMoves.get(0);

            int i;
            for (i = 0; i < numRootMoves; ++i) {
                Context copyContext = this.copyContext(context);
                Move m = (Move) this.sortedRootMoves.get(i);
                game.apply(copyContext, m);
                float value = this.alphaBeta(copyContext, searchDepth - 1, alpha, beta, maximisingPlayer, stopTime);
                if (System.currentTimeMillis() >= stopTime || this.wantsInterrupt) {
                    bestMove = null;
                    break;
                }

                int origMoveIdx = this.currentRootMoves.indexOf(m);
                if (origMoveIdx >= 0) {
                    this.rootValueEstimates.set(origMoveIdx, (float) this.scoreToValueEst(value, this.rootAlphaInit, this.rootBetaInit));
                }

                moveScores.set(i, value);
                if (maximisingPlayer == 1) {
                    // When new best move found, store move
                    if (value > score) {
                        score = value;
                        bestMove = m;
                    }

                    // When new lower bound found, store lower bound
                    if (score > alpha) {
                        alpha = score;
                    }

                    // Alpha Beta cut-off
                    if (alpha >= beta) {
                        break;
                    }
                } else {
                    // When new best move found, store move
                    if (value < score) {
                        bestMove = m;
                        score = value;
                    }

                    // When new upper bound found, store upper bound
                    if (score < beta) {
                        beta = score;
                    }

                    // Alpha Beta cut-off
                    if (alpha >= beta) {
                        break;
                    }
                }
            }

            // alpha-beta is over, this is iterative deepening stuff again

            if (bestMove != null) {
                this.estimatedRootScore = score;
                if ((score == this.rootBetaInit && maximisingPlayer == 1) || (score == this.rootAlphaInit && maximisingPlayer == 2)) {
                    // we've just proven a win, so we can return best move
                    // found during this search
                    this.analysisReport = this.friendlyName + " (player " + maximisingPlayer + ") found a proven win at depth " + searchDepth + ".";
                    this.provedWin = true;
                    return bestMove;
                }
                if ((score == this.rootAlphaInit && maximisingPlayer == 1) || (score == this.rootBetaInit && maximisingPlayer == 2)) {
                    // we've just proven a loss, so we return the best move
                    // of the PREVIOUS search (delays loss for the longest
                    // amount of time)
                    this.analysisReport = this.friendlyName + " (player " + maximisingPlayer + ") found a proven loss at depth " + searchDepth + ".";
                    return bestMoveCompleteSearch;
                }

                if (this.searchedFullTree) {
                    // We've searched full tree but did not prove a win or loss
                    // probably means a draw, play best line we have
                    this.analysisReport = this.friendlyName + " (player " + maximisingPlayer + ") completed search of depth " + searchDepth + " (no proven win or loss).";
                    return bestMove;
                }

                bestMoveCompleteSearch = bestMove;
            } else {
                // decrement because we didn't manage to complete this search
                searchDepth -= searchDepthIncrement;
            }

            if (System.currentTimeMillis() >= stopTime || this.wantsInterrupt) {
                this.analysisReport = this.friendlyName + " (player " + maximisingPlayer + ") completed search of depth " + searchDepth + ".";
                return bestMoveCompleteSearch;
            }

            // order moves based on scores found, for next search
            scoredMoves.clear();
            for (i = 0; i < numRootMoves; ++i) {
                scoredMoves.add(new ScoredMove((Move) this.sortedRootMoves.get(i), moveScores.get(i)));
            }

            if (maximisingPlayer == 1) {
                Collections.sort(scoredMoves);
            } else {
                Collections.sort(scoredMoves, Collections.reverseOrder());
            }
            this.sortedRootMoves.clear();

            for (i = 0; i < numRootMoves; ++i) {
                this.sortedRootMoves.add(((ScoredMove) scoredMoves.get(i)).move);
            }

            // clear the vector of scores
            moveScores.fill(0, numRootMoves, 0.0F);
        }

        this.analysisReport = this.friendlyName + " (player " + maximisingPlayer + ") completed search of depth " + searchDepth + ".";
        return bestMoveCompleteSearch;
    }

    /**
     * Recursive alpha-beta search function, while also store the storing the trainings data found after searching
     *
     * @param context Copy of the context containing the current state of the game
     * @param depth Current search depth
     * @param inAlpha Current lower bound of search
     * @param inBeta Current upper bound of search
     * @param maximisingPlayer Who is the maximising player?
     * @param stopTime Time to terminate the search
     * @return evaluation of the reached state, from perspective of maximising player.
     */
    public float alphaBeta(Context context, int depth, float inAlpha, float inBeta, int maximisingPlayer, long stopTime) {
        // Add iteration
        iterations++;

        Trial trial = context.trial();
        State state = context.state();
        float alpha = inAlpha;
        float beta = inBeta;
        long zobrist = state.fullHash(context);
        TranspositionTable.ABTTData tableData;
        if (this.transpositionTable != null) {
            tableData = this.transpositionTable.retrieve(zobrist);
            if (tableData != null && tableData.depth >= depth) {
                // Already searched deep enough for data in TT, use results
                switch (tableData.valueType) {
                    case 1:
                        return tableData.value;
                    case 2:
                        alpha = Math.max(inAlpha, tableData.value);
                        break;
                    case 3:
                    default:
                        System.err.println("INVALID TRANSPOSITION TABLE DATA!");
                        break;
                    case 4:
                        beta = Math.min(inBeta, tableData.value);
                }

                if (alpha >= beta) {
                    return tableData.value;
                }
            }
        } else {
            tableData = null;
        }

        if (!trial.over() && context.active(maximisingPlayer)) {
            int numLegalMoves;
            if (depth == 0) {
                // non-terminal leaf evaluation
                this.searchedFullTree = false;
                float heuristicScore = this.leafEvaluator.evaluate(context, 1);

                this.minHeuristicEval = Math.min(this.minHeuristicEval, heuristicScore);
                this.maxHeuristicEval = Math.max(this.maxHeuristicEval, heuristicScore);
                return heuristicScore;
            } else {
                Game game = context.game();
                int mover = state.playerToAgent(state.mover());
                FastArrayList<Move> legalMoves = game.moves(context).moves();
                numLegalMoves = legalMoves.size();
                if (tableData != null) {
                    // Put best move according to Transposition Table first
                    Move transpositionBestMove = tableData.bestMove;
                    legalMoves = new FastArrayList(legalMoves);

                    for (int i = 0; i < numLegalMoves; ++i) {
                        if (transpositionBestMove.equals(legalMoves.get(i))) {
                            Move temp = (Move) legalMoves.get(0);
                            legalMoves.set(0, (Move) legalMoves.get(i));
                            legalMoves.set(i, temp);
                            break;
                        }
                    }
                }

                Move bestMove = (Move) legalMoves.get(0);
                int i;
                Context copyContext = null;
                Move m;
                float value;
                float score;
                if (mover == 1) {
                    score = -1000000.0F;
                    i = 0;

                    while (true) {
                        if (i < numLegalMoves) {
                            copyContext = this.copyContext(context);
                            m = (Move) legalMoves.get(i);
                            game.apply(copyContext, m);
                            value = this.alphaBeta(copyContext, depth - 1, alpha, beta, maximisingPlayer, stopTime);
                            if (System.currentTimeMillis() >= stopTime || this.wantsInterrupt) {
                                return 0.0F;
                            }

                            if (value > score) {
                                bestMove = m;
                                score = value;
                            }

                            if (score > alpha) {
                                alpha = score;
                            }

                            if (!(alpha >= beta)) {
                                ++i;
                                continue;
                            }
                        }

                        if (this.transpositionTable != null) {
                            if (score <= inAlpha) {
                                this.transpositionTable.store(bestMove, zobrist, score, depth, (byte) 4);
                            } else if (score >= beta) {
                                this.transpositionTable.store(bestMove, zobrist, score, depth, (byte) 2);
                            } else {
                                this.transpositionTable.store(bestMove, zobrist, score, depth, (byte) 1);
                            }

                            if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                                this.TTTraining.store(zobrist, score, depth,
                                        this.leafEvaluator.boardToInput(context).data().asFloat());
                            }
                        }

                        return score;
                    }
                } else {
                    score = 1000000.0F;
                    i = 0;

                    while (true) {
                        if (i < numLegalMoves) {
                            copyContext = this.copyContext(context);
                            m = (Move) legalMoves.get(i);
                            game.apply(copyContext, m);
                            value = this.alphaBeta(copyContext, depth - 1, alpha, beta, maximisingPlayer, stopTime);
                            if (System.currentTimeMillis() >= stopTime || this.wantsInterrupt) {
                                return 0.0F;
                            }

                            if (value < score) {
                                bestMove = m;
                                score = value;
                            }

                            if (score < beta) {
                                beta = score;
                            }

                            if (!(alpha >= beta)) {
                                ++i;
                                continue;
                            }
                        }

                        if (this.transpositionTable != null) {
                            if (score <= inAlpha) {
                                this.transpositionTable.store(bestMove, zobrist, score, depth, (byte) 4);
                            } else if (score >= beta) {
                                this.transpositionTable.store(bestMove, zobrist, score, depth, (byte) 2);
                            } else {
                                this.transpositionTable.store(bestMove, zobrist, score, depth, (byte) 1);
                            }

                            if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                                this.TTTraining.store(zobrist, score, depth,
                                        this.leafEvaluator.boardToInput(context).data().asFloat());
                            }
                        }

                        return score;
                    }
                }
            }
        } else {
            // Terminal evaluation
            float score = this.terminalEvaluator.evaluate(context, 1);
            if (depth == 0 || this.dataSelection == Enums.DataSelection.TREE){
                this.TTTraining.store(zobrist, score, depth,
                        this.leafEvaluator.boardToInput(context).data().asFloat());
            }
            return score;
        }
    }

    /**
     * Converts a score into a value estimate in [-1, 1]. Useful for visualisations.
     *
     * @param score Score found by AlphaBetaSearch
     * @param alpha Lower bound of search
     * @param beta Upper bound of search
     * @return Value estimate in [-1, 1] from unbounded (heuristic) score.
     */
    public double scoreToValueEst(float score, float alpha, float beta) {
        if (score == alpha) {
            return -1.0;
        } else {
            return score == beta ? 1.0 : -0.8 + 1.6 * (double) ((score - this.minHeuristicEval) / (this.maxHeuristicEval - this.minHeuristicEval));
        }
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Set the playerID, initialise a new Transposition Table and initialise both GameStateEvaluators
     *
     * @param game The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    public void initAI(Game game, int playerID) {
        this.estimatedRootScore = 0.0F;
        this.maxHeuristicEval = 0.0F;
        this.minHeuristicEval = 0.0F;
        this.analysisReport = null;
        this.currentRootMoves = null;
        this.rootValueEstimates = null;
        this.lastSearchedRootContext = null;
        this.lastReturnedMove = null;
        this.numPlayersInGame = game.players().count();
        if (!game.usesNoRepeatPositionalInGame() && !game.usesNoRepeatPositionalInTurn()) {
            if (!this.allowTranspositionTable) {
                this.transpositionTable = null;
            } else {
                this.transpositionTable = new TranspositionTable(12);
            }
        } else {
            this.transpositionTable = null;
        }

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, this.net);
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();


    }

    /**
     * Allows an agent to tell Ludii whether or not it can support playing
     * any given game. Copied from the AlphaBeta implementation from Ludii.
     *
     * @param game Ludii's game
     * @return False if the AI cannot play the given game.
     */
    public boolean supportsGame(Game game) {
        if (game.players().count() <= 1 || game.players().count() > 2) {
            return false;
        } else if (game.hiddenInformation()) {
            return false;
        } else {
            return game.hasSubgames() ? false : game.isAlternatingMoveGame();
        }
    }

    /**
     * Calls the scoreToValueEst function for the current root node
     * @return A value estimate in [-1, 1] for the current root node
     */
    public double estimateValue() {
        return this.scoreToValueEst(this.estimatedRootScore, this.rootAlphaInit, this.rootBetaInit);
    }

    /**
     * Getter for the analysis report
     * @return String representing the analysis report
     */
    public String generateAnalysisReport() {
        return this.analysisReport;
    }

    /**
     * Creates visualisation data for Ludii
     * @return visualisation data
     */
    public AIVisualisationData aiVisualisationData() {
        if (this.currentRootMoves != null && this.rootValueEstimates != null) {
            FVector aiDistribution = this.rootValueEstimates.copy();
            aiDistribution.subtract(aiDistribution.min());
            return new AIVisualisationData(aiDistribution, this.rootValueEstimates, this.currentRootMoves);
        } else {
            return null;
        }
    }

    /**
     * Wrapper for score + move, used for sorting moves based on scores.
     *
     * @author Dennis Soemers
     */
    protected class ScoredMove implements Comparable<ScoredMove> {
        /** The move */
        public final Move move;
        /** The move's score */
        public final float score;

        /**
         * Constructor
         * @param move
         * @param score
         */
        public ScoredMove(Move move, float score) {
            this.move = move;
            this.score = score;
        }

        public int compareTo(ScoredMove other) {
            float delta = other.score - this.score;
            if (delta < 0.0F) {
                return -1;
            } else {
                return delta > 0.0F ? 1 : 0;
            }
        }
    }

    /**
     * Enum for allowed search depth
     */
    public static enum AllowedSearchDepths {
        Any,
        Even,
        Odd;

        private AllowedSearchDepths() {
        }
    }
}
