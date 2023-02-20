package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCTDifferenceV;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS search algorithm using neural networks with a modified Implicit UCT (using difference to mean instead of
 * estimated value), no play-outs, while backpropagating the estimated value by the network, and robust child.
 */
public class ImplicitMCTSNNNP16 extends MCTS {

    //-------------------------------------------------------------------------

    /** Path to the neural network */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * (influence estimated value = 0.8, exploration=0.001, QInit=DRAW, 4 threads)
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSNNNP16(String pathName) {
        super(new ImplicitUCTDifferenceV(.8, .001f), new RandomPlayout(0),
                new FixedEarlyTerminationBackprop(), new RobustChild());
//                new IterationWrapper(new RobustChild()));
//                new debugFinalSelectionWrapper(new RobustChild()));

        this.pathName = pathName;

        this.setNumThreads(4);
        this.setQInit(QInit.DRAW);
    }

    /**
     * Constructor with the path to the desired neural network as string, the influence of the estimated value,
     * and exploration as input. (QInit=DRAW, 4 threads)
     * @param pathName Path to the desired neural network
     * @param alpha influence of the estimated value
     * @param explorationConstant exploration constant
     */
    public ImplicitMCTSNNNP16(String pathName, float alpha, float explorationConstant) {
        super(new ImplicitUCTDifferenceV(alpha, explorationConstant), new RandomPlayout(0),
                new FixedEarlyTerminationBackprop(), new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
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