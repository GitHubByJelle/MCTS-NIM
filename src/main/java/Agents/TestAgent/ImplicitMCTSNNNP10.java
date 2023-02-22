package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkTTLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCT;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS search algorithm using neural networks and transposition tables with Implicit UCT, no play-outs, while
 * backpropagating the estimated value by the network, and robust child.
 */
public class ImplicitMCTSNNNP10 extends MCTS {
    //-------------------------------------------------------------------------

    /**
     * Path to the neural network
     */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (influence estimated value = 0.8, exploration=0.001, QInit=PARENT, 4 threads)
     *
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNNP10(String pathName) {
        super(new ImplicitUCT(.8, .001f), new RandomPlayout(0),
                new FixedEarlyTerminationBackprop(), new RobustChild());

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

        this.setLeafEvaluator(new MultiNeuralNetworkTTLeafEvaluator(game,
                LearningManager.loadNetwork(pathName, false), this.numThreads), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}