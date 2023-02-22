package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCTGRAVE;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * Combination of Implicit MCTS, GRAVE using a NN with no play-outs.
 */
public class ImplicitMCTSGRAVENN_NP extends MCTS {

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
    public ImplicitMCTSGRAVENN_NP(String pathName) {
        super(new ImplicitUCTGRAVE(100, 10.e-6, 0.8, Math.sqrt(2.0)),
                new RandomPlayout(0),
                new FixedEarlyTerminationBackprop(),
                new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
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