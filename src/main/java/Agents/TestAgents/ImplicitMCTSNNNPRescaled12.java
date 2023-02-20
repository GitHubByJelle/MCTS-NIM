package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Rescaler.MultiplyDifferences;
import MCTSStrategies.Selection.ImplicitUCTRescaled;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS search algorithm using neural networks with Implicit UCT, no play-outs, while
 * backpropagating the estimated value by the network, and robust child. Instead of using the estimated
 * values directly, the estimated values get rescaled by multiplying the difference to the mean.
 */
public class ImplicitMCTSNNNPRescaled12 extends MCTS {

    //-------------------------------------------------------------------------

    /** Path to the neural network */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (influence estimated value = 0.8, exploration=0.001, multiplier=2, QInit=DRAW, 12 threads)
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNNPRescaled12(String pathName) {
        super(new ImplicitUCTRescaled(.8, .001f, new MultiplyDifferences(2)),
                new RandomPlayout(0),new FixedEarlyTerminationBackprop(), new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(12);
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
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}