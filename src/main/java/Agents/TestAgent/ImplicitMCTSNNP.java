package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Playout.EpsilonGreedyPlayout;
import MCTSStrategies.Selection.ImplicitUCT;
import Training.LearningManager;
import game.Game;
import search.mcts.backpropagation.MonteCarloBackprop;

/**
 * MCTS search algorithm using neural networks with Implicit UCT, epsilon-greedy play-outs, while
 * backpropagating the estimated value by the network, and robust child.
 */
public class ImplicitMCTSNNP extends MCTS {

    //-------------------------------------------------------------------------

    /**
     * Path to the neural network
     */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (solver=true, influence estimated value = 0.8, exploration=sqrt(2), epsilon=0.05, QInit=PARENT, 4 threads)
     *
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNP(String pathName) {
        super(new ImplicitUCT(.8, Math.sqrt(2)), new EpsilonGreedyPlayout(.05f, -1),
                new MonteCarloBackprop(), new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
        this.setQInit(QInit.PARENT);
    }

    /**
     * Constructor with the path to the desired neural network as string, the influence of the estimated value,
     * and exploration as input. (solver=true, epsilon=0.05, QInit=PARENT, 4 threads)
     *
     * @param pathName            Path to the desired neural network
     * @param alpha               influence of the estimated value
     * @param explorationConstant exploration constant
     */
    public ImplicitMCTSNNP(String pathName, float alpha, float explorationConstant) {
        super(new ImplicitUCT(alpha, explorationConstant), new EpsilonGreedyPlayout(.05f, -1),
                new MonteCarloBackprop(), new RobustChild());

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