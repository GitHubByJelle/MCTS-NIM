package Agents.TestAgent;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MSLeafEvaluator;
import Evaluator.TanhEvaluatorWrapper;
import game.Game;
import utils.data_structures.transposition_table.TranspositionTable;

/**
 * Implementation of Ludii's alpha-beta search using the Maarten Schadds evaluation function.
 */
public class AlphaBetaSearchMS extends AlphaBetaSearchNN {

    /**
     * Constructor with no inputs
     */
    public AlphaBetaSearchMS() {
        this.friendlyName = "AlphaBeta with Maarten Schadds evaluation function";
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

        this.leafEvaluator = new TanhEvaluatorWrapper(new MSLeafEvaluator(game),
                60, 100, -100);
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
    }
}
