package Agents.MCTSTest;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MSLeafEvaluator;
import Evaluator.ParallelNeuralNetworkLeafEvaluator;
import Evaluator.TanhEvaluatorWrapper;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCTGRAVE;
import Training.LearningManager;
import game.Game;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.RandomPlayout;
import search.mcts.selection.UCB1;

/**
 * Combination of Implicit MCTS, GRAVE using a NN with random play-outs.
 */
public class ImplicitMCTSGRAVENN_Random extends MCTS {

    //-------------------------------------------------------------------------

    /** Path to the neural network */
    String pathName;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the path to the desired neural network as string
     * @param pathName Path to the desired neural network
     */
    public ImplicitMCTSGRAVENN_Random(String pathName) {
        super(new ImplicitUCTGRAVE(100, 10.e-6, 0.8, Math.sqrt(2.0)),
                new RandomPlayout(-1),
                new MonteCarloBackprop(),
                new RobustChild());
        this.pathName = pathName;

        this.setNumThreads(6);
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

        this.setLeafEvaluator(new ParallelNeuralNetworkLeafEvaluator(game,
                LearningManager.loadNetwork(pathName, false), 64), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}