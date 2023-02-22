package Agents.TestAgent;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.RandomLeafEvaluator;
import game.Game;
import utils.data_structures.transposition_table.TranspositionTable;

/**
 * Implementation of Ludii's alpha-beta search using pseudorandom number generator as evaluation function.
 */
public class AlphaBetaSearchRandom extends AlphaBetaSearchNN {

    /**
     * Constructor with no inputs
     */
    public AlphaBetaSearchRandom() {
        this.friendlyName = "AlphaBeta with Random eval";
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Sets all needed variables for alpha beta search w.r.t. to the GameStateEvaluators useds
     *
     * @param game     The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    @Override
    public void initAI(final Game game, final int playerID) {
        this.rootAlphaInit = -1;
        this.rootBetaInit = 1;
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

        this.leafEvaluator = new RandomLeafEvaluator();
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
    }
}
