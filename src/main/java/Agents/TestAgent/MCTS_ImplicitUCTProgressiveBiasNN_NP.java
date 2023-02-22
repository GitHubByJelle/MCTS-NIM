package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.InitialNoPlayoutTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCTProgressiveBias;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * Combination of Implicit MCTS and Progressive Bias using a NN.
 */
public class MCTS_ImplicitUCTProgressiveBiasNN_NP extends MCTS {

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
    public MCTS_ImplicitUCTProgressiveBiasNN_NP(String pathName) {
        super(new ImplicitUCTProgressiveBias(0.8f, 0.01f),
                new RandomPlayout(0),
                new InitialNoPlayoutTerminationBackprop(),
                new RobustChild());

        this.pathName = pathName;

        this.friendlyName = "MCTS-Test";
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