package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MSLeafEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import Evaluator.TanhEvaluatorWrapper;
import MCTSStrategies.Backpropagation.DynamicEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Playout.DynamicEpsilonGreedyPlayout;
import MCTSStrategies.Selection.ImplicitUCTAlphaDecrease;
import Training.LearningManager;
import game.Game;
import other.GameLoader;

/**
 * MCTS search algorithm using neural networks with enhanced Implicit UCT, epsilon-greedy play-out with dynamic
 * early termination (when |x| > bound, backpropagate win or loss), and robust child. nstead of using the same value for alpha
 * * the entire search, the alpha value decreases when the number of visits increases with a specified slope
 */
public class ImplicitMCTSNNMSP4 extends MCTS {

    //-------------------------------------------------------------------------

    /**
     * Path to the neural network
     */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (initial influence estimated value = 0.8, exploration=0.0001, slope=0.05, QInit=PARENT, 4 threads)
     *
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNMSP4(String pathName) {
        super(new ImplicitUCTAlphaDecrease(.8, .0001, 0.05f),
                new DynamicEpsilonGreedyPlayout(.1f, .4f,
                        new TanhEvaluatorWrapper(new MSLeafEvaluator(GameLoader.loadGameFromName("Breakthrough.lud")),
                                60, 100, -100),
                        new ClassicTerminalStateEvaluator()),
                new DynamicEarlyTerminationBackprop(.4f,
                        new TanhEvaluatorWrapper(new MSLeafEvaluator(GameLoader.loadGameFromName("Breakthrough.lud")),
                                60, 100, -100)),
                new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(8);
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