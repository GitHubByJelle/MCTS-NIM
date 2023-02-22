package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.RandomLeafEvaluator;
import game.Game;
import utils.Enums;
import utils.TranspositionTableStamp;

/**
 * Selects the best move to play based by using pseudorandom number generator as evaluation function in
 * combination with descent. Has been used as debug check.
 */
public class descentRandom extends descent {

    /**
     * Constructor with no inputs (uses epsilon-greedy exploration policy and safest selection policy).
     */
    public descentRandom() {
        this.friendlyName = "Descent (Random evaluation)";
        this.explorationPolicy = Enums.ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = Enums.SelectionPolicy.SAFEST;
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

        this.leafEvaluator = new RandomLeafEvaluator();
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
    }
}
