package Agents.TestAgent;

import Agents.MCTS;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.ParallelNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ProgressiveHistoryGRAVE;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS implementation with Progressive History and fixed early termination using NNs
 */
public class MCTS_ProgessiveHistoryNN_FET extends MCTS {

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
    public MCTS_ProgessiveHistoryNN_FET(String pathName) {
        // Original
        super(new ProgressiveHistoryGRAVE(100, 10.e-6, 3, Math.sqrt(2.0)),
                new RandomPlayout(4),
                new FixedEarlyTerminationBackprop(),
                new RobustChild());

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
                LearningManager.loadNetwork(pathName, false), 6), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}