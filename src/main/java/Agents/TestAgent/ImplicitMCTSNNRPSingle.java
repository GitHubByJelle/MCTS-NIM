package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.ParallelNeuralNetworkLeafEvaluator;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCT;
import Training.LearningManager;
import game.Game;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS search algorithm using neural networks (parallel inference) with Implicit UCT, random play-outs, while
 * backpropagating the estimated value by the network, and robust child with solver.
 */
public class ImplicitMCTSNNRPSingle extends MCTS {

    //-------------------------------------------------------------------------

    /**
     * Path to the neural network
     */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (solver=true, influence estimated value = 0.8, exploration=sqrt(2), 1 thread)
     *
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNRPSingle(String pathName) {
        super(new ImplicitUCT(.8, Math.sqrt(2)), new RandomPlayout(-1),
                new MonteCarloBackprop(), new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(1);
        this.setUseSolver(true);
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