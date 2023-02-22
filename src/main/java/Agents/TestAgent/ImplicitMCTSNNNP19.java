package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCTAlphaDecrease;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS search algorithm using neural networks with enhanced Implicit UCT, no play-outs, while
 * backpropagating the estimated value by the network, and robust child. Instead of using the same value for alpha
 * the entire search, the alpha value decreases when the number of visits increases with a specified slope
 */
public class ImplicitMCTSNNNP19 extends MCTS {

    //-------------------------------------------------------------------------

    /**
     * Path to the neural network
     */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (initial influence estimated value = 0.8, exploration=0.001, slope=0.05, QInit=PARENT, 4 threads)
     *
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNNP19(String pathName) {
        super(new ImplicitUCTAlphaDecrease(.8, .001f, .05f),
                new RandomPlayout(0),
                new FixedEarlyTerminationBackprop(),
                new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
        this.setQInit(QInit.PARENT);
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

        this.setLeafEvaluator(new MultiNeuralNetworkLeafEvaluator(game,
                LearningManager.loadNetwork(pathName, false), this.numThreads), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}