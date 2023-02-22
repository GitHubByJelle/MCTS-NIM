package Agents;

import Agents.TestAgent.AlphaBetaSearchNN;
import Evaluator.HeuristicLeafEvaluator;
import Evaluator.MaxClassicTerminalStateEvaluator;
import game.Game;
import utils.data_structures.transposition_table.TranspositionTable;

/**
 * Implementation of alpha-beta search using the Ludii's evaluation function. Is similar to Ludii's implementation.
 */
public class AlphaBetaSearchHF extends AlphaBetaSearchNN {

    /**
     * Constructor with no inputs
     */
    public AlphaBetaSearchHF() {
        this.friendlyName = "AlphaBeta with HF";
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
        this.rootAlphaInit = -9999999;
        this.rootBetaInit = 9999999;
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

        this.leafEvaluator = new HeuristicLeafEvaluator(game);
        this.terminalEvaluator = new MaxClassicTerminalStateEvaluator();
    }
}
