package Agents;

import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Playout.EpsilonGreedyPlayout;
import MCTSStrategies.Selection.ImplicitUCT;
import MCTSStrategies.Wrapper.TrainingPlayoutWrapper;
import MCTSStrategies.Wrapper.TrainingSelectionWrapper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import search.mcts.backpropagation.MonteCarloBackprop;

/**
 * MCTS implementation for training with the descent framework, using Implicit UCT, epsilon-greedy play-outs,
 * Monte Carlo backpropagation and robust child (inspired by descent).
 */
public class ImplicitMCTSNNTraining1 extends MCTSTraining {

    /**
     * Constructor with MultiLayerNetwork (DeepLearning4J) as input (exploration=0.01, influence estimated value = 0.8,
     * epsilon = 0.05, and 4 threads)
     */
    public ImplicitMCTSNNTraining1(MultiLayerNetwork net) {
        super(new ImplicitUCT(.8, .0001),
                new TrainingPlayoutWrapper(new EpsilonGreedyPlayout(.05f, -1)),
                new MonteCarloBackprop(),
                new TrainingSelectionWrapper(new RobustChild()), net);

        this.setNumThreads(1);
    }
}