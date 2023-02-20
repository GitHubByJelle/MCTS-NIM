package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Rescaler.Softmax;
import MCTSStrategies.Selection.ImplicitUCTAlphaDecreaseRescaledExploration;
import MCTSStrategies.Selection.ImplicitUCTBoundedAlphaDecreaseRescaledExploration;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS search algorithm using neural networks with enhanced Implicit UCT, no play-outs, while
 * backpropagating the estimated value by the network, and robust child. Instead of using the same value for alpha
 * and exploration the entire search, the alpha value decreases when the number of visits increases with a specified
 * slope until a minimum bound is reached, while the exploration value gets determined with a softmax (with temperature) based on the implicit minimax
 * values.
 */
public class ImplicitMCTSNNNP42 extends MCTS {

    //-------------------------------------------------------------------------

    /** Path to the neural network */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (initial influence estimated value = 0.8, exploration=2, slope=0.05, minimum influence of estimated values = 0.1,
     * QInit=PARENT, 4 threads)
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNNP42(String pathName) {
        super(new ImplicitUCTBoundedAlphaDecreaseRescaledExploration(.8, 2,
                        new Softmax(), 1/20f, .1),
                new RandomPlayout(0),
                new FixedEarlyTerminationBackprop(), new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
        this.setQInit(QInit.PARENT);
    }

    /**
     * Constructor with the path to the desired neural network as string, the influence of the estimated value,
     * exploration value, and slope as input. (QInit=PARENT, 4 threads)
     * @param pathName Path to the desired neural network
     * @param alpha influence of the estimated value
     * @param explorationConstant exploration constant
     */
    public ImplicitMCTSNNNP42(String pathName, float alpha, float explorationConstant, float slope) {
        super(new ImplicitUCTAlphaDecreaseRescaledExploration(alpha, explorationConstant,
                        new Softmax(), slope),
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