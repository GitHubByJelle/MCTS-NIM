package Agents.MCTSTest;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MSLeafEvaluator;
import Evaluator.TanhEvaluatorWrapper;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ProgressiveBiasGRAVE;
import game.Game;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.MAST;

/**
 * MCTS implementation with Progressive Bias, GRAVE and MAST with solver, using the Maarten Schadds evaluation function
 */
public class MCTS_ProgressiveBiasGRAVE_MAST_Solver extends MCTS {
    /**
     * Constructor with no inputs
     */
    public MCTS_ProgressiveBiasGRAVE_MAST_Solver() {
        // Original
        super(new ProgressiveBiasGRAVE(),
                new MAST(-1, .05f),
                new MonteCarloBackprop(),
                new RobustChild());

        this.friendlyName = "MCTS-Test";
        this.setNumThreads(6);
        this.setUseSolver(true);
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Initialise the parent and both GameStateEvaluators
     *
     * @param game The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    public void initAI(Game game, int playerID){
        super.initParent(game, playerID);

        this.setLeafEvaluator(new TanhEvaluatorWrapper(new MSLeafEvaluator(game),
                60, 100, -100), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}