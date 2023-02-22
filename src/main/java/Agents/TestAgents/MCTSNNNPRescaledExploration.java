package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Rescaler.Softmax;
import MCTSStrategies.Selection.ImplicitUCTRescaledExploration6;
import MCTSStrategies.Selection.UCTRescaledExploration;
import Training.LearningManager;
import game.Game;
import search.mcts.playout.RandomPlayout;

// Based on ImplicitMCTSNNNPRescaledExploration8
public class MCTSNNNPRescaledExploration extends MCTS {
    String pathName;

    public MCTSNNNPRescaledExploration(String pathName) {
        super(new UCTRescaledExploration(2, new Softmax()),
                new RandomPlayout(0),
                new FixedEarlyTerminationBackprop(), new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
//        this.setUseSolver(true);
        this.setQInit(QInit.PARENT);
    }

    public MCTSNNNPRescaledExploration(String pathName, float alpha, float explorationConstant) {
        super(new ImplicitUCTRescaledExploration6(alpha, new Softmax()), new RandomPlayout(0),
                new FixedEarlyTerminationBackprop(), new RobustChild());

        this.pathName = pathName;

        this.setNumThreads(4);
//        this.setUseSolver(true);
        this.setQInit(QInit.PARENT);
    }

    public void initAI(Game game, int playerID) {
        super.initParent(game, playerID);

        this.setLeafEvaluator(new MultiNeuralNetworkLeafEvaluator(game,
                LearningManager.loadNetwork(pathName, false), this.numThreads), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}