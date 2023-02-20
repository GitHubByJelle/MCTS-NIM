package Agents.MCTSTest;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import Evaluator.ParallelNeuralNetworkLeafEvaluator;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCT;
import MCTSStrategies.Selection.ImplicitUCTGRAVE;
import Training.LearningManager;
import game.Game;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.MAST;

/**
 * Implicit MCTS using a NN with MAST.
 */
public class ImplicitMCTSNN_MAST extends MCTS {

    //-------------------------------------------------------------------------

    /** Path to the neural network */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNN_MAST(String pathName) {
        super(new ImplicitUCT(.8f, Math.sqrt(2)),
                new MAST(-1, .05f),
                new MonteCarloBackprop(),
                new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
        this.setQInit(QInit.PARENT);
    }

    /**
     * Constructor with the path to the desired neural network as string, the influence of the
     * estimated value and the exploration constant as input
     * @param pathName Path to the desired neural network
     * @param alpha influence of estimated value
     * @param explorationConstant exploration constant
     */
    public ImplicitMCTSNN_MAST(String pathName, float alpha, float explorationConstant) {
        super(new ImplicitUCT(alpha, explorationConstant),
                new MAST(-1, .05f),
                new MonteCarloBackprop(),
                new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
        this.setQInit(QInit.PARENT);
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Initialise the parent and both GameStateEvaluators
     *
     * @param game The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    public void initAI(Game game, int playerID) {
        super.initParent(game, playerID);

        this.setLeafEvaluator(new MultiNeuralNetworkLeafEvaluator(game,
                LearningManager.loadNetwork(pathName, false), this.numThreads), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}