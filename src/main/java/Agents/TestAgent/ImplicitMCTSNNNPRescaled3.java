package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.ParallelNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Rescaler.MultiplyDifferences;
import MCTSStrategies.Selection.ImplicitUCTRescaled;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS search algorithm using neural networks (parallel inference) with Implicit UCT, no play-outs, while
 * backpropagating the estimated value by the network, and robust child. Instead of using the estimated
 * values directly, the estimated values get rescaled by multiplying the difference to the mean.
 */
public class ImplicitMCTSNNNPRescaled3 extends MCTS {

    //-------------------------------------------------------------------------

    /**
     * Path to the neural network
     */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (influence estimated value = 0.8, exploration=0.01, multiplier=2, QInit=DRAW, 6 threads)
     *
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNNPRescaled3(String pathName) {
        super(new ImplicitUCTRescaled(.8, .01f, new MultiplyDifferences(2)),
                new RandomPlayout(0), new FixedEarlyTerminationBackprop(), new RobustChild());

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