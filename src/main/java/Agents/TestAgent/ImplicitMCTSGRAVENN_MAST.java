package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.ParallelNeuralNetworkLeafEvaluator;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCTGRAVE;
import Training.LearningManager;
import game.Game;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.MAST;

/**
 * Combination of Implicit MCTS, GRAVE using a NN with MAST.
 */
public class ImplicitMCTSGRAVENN_MAST extends MCTS {

    //-------------------------------------------------------------------------

    /**
     * Path to the neural network
     */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     *
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSGRAVENN_MAST(String pathName) {
        super(new ImplicitUCTGRAVE(100, 10.e-6, 0.8, Math.sqrt(2.0)),
                new MAST(-1, .05f),
                new MonteCarloBackprop(),
                new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(6);
        this.setQInit(QInit.DRAW);
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

        this.setLeafEvaluator(new ParallelNeuralNetworkLeafEvaluator(game,
                LearningManager.loadNetwork(pathName, false), 64), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}