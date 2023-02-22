package Agents;

import Agents.TestAgent.MCTSTraining;
import MCTSStrategies.Backpropagation.InitialNoPlayoutTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Selection.ImplicitUCT;
import MCTSStrategies.Wrapper.TrainingSelectionWrapper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS implementation for training with the descent framework, using Implicit UCT, no play-outs,
 * estimated value backpropagation and robust child (inspired by UBFM).
 */
public class ImplicitMCTSNNTraining2 extends MCTSTraining {

    /**
     * Constructor with MultiLayerNetwork (DeepLearning4J) as input (exploration=0.01, influence estimated value = 0.8,
     * epsilon = 0.05, and 4 threads)
     */
    public ImplicitMCTSNNTraining2(MultiLayerNetwork net) {
        super(new ImplicitUCT(.8, .0001),
                new RandomPlayout(0),
                new InitialNoPlayoutTerminationBackprop(),
                new TrainingSelectionWrapper(new RobustChild()), net);

        this.setNumThreads(4);
    }
}