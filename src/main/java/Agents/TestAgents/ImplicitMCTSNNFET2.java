package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import Evaluator.ParallelNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Playout.EpsilonGreedyPlayout;
import MCTSStrategies.Selection.ImplicitUCT;
import Training.LearningManager;
import game.Game;

/**
 * MCTS search algorithm using neural networks with Implicit UCT, epsilon-greedy play-out with fixed early termination
 * (backpropagation the estimated value by the network), and robust child.
 */
public class ImplicitMCTSNNFET2 extends MCTS {

    //-------------------------------------------------------------------------

    /** Path to the neural network */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (solver=true, influence estimated value = 0.8, exploration=0.001, epsilon=0.05 QInit=DRAW, 4 threads)
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNFET2(String pathName) {
        super(new ImplicitUCT(.8, 0.001f), new EpsilonGreedyPlayout(.05f, 4),
                new FixedEarlyTerminationBackprop(), new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
        this.setUseSolver(true);
        this.setQInit(QInit.DRAW);
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
        this.setTerminalStateEvaluator( new ClassicTerminalStateEvaluator());
    }
}