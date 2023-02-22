package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MSLeafEvaluator;
import Evaluator.TanhEvaluatorWrapper;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ProgressiveBiasEvaluator;
import game.Game;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.MAST;

/**
 * MCTS implementation with Progressive Bias and MAST, using the Maarten Schadds evaluation function
 */
public class MCTS_ProgressiveBias_MAST extends MCTS {
    /**
     * Constructor with no inputs
     */
    public MCTS_ProgressiveBias_MAST() {
        // Original
        super(new ProgressiveBiasEvaluator(),
                new MAST(-1, .05f),
                new MonteCarloBackprop(),
                new RobustChild());

        this.friendlyName = "MCTS-Test";
        this.setNumThreads(16);
        ;
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Initialise the parent and both GameStateEvaluators
     *
     * @param game     The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    public void initAI(Game game, int playerID) {
        super.initParent(game, playerID);

        this.setLeafEvaluator(new TanhEvaluatorWrapper(new MSLeafEvaluator(game),
                60, 100, -100), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}