package Agents.TestAgent;

import MCTSStrategies.Backpropagation.InitialNoPlayoutTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Rescaler.Softmax;
import MCTSStrategies.Selection.ImplicitUCTBoundedAlphaDecreaseRescaledExploration;
import MCTSStrategies.Wrapper.TrainingSelectionWrapper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS implementation for training with the descent framework, using enhanced Implicit UCT, no play-outs,
 * estimated value backpropagation and robust child (inspired by UBFM).
 */
public class ImplicitMCTSNNTraining4 extends MCTSTraining {

    /**
     * Constructor with MultiLayerNetwork (DeepLearning4J) as input (exploration=0.01, influence estimated value = 0.8,
     * and 4 threads)
     */
    public ImplicitMCTSNNTraining4(MultiLayerNetwork net) {
        super(new ImplicitUCTBoundedAlphaDecreaseRescaledExploration(0.6, 2,
                        new Softmax(), 0.1f, 0.3),
                new RandomPlayout(0),
                new InitialNoPlayoutTerminationBackprop(),
                new TrainingSelectionWrapper(new RobustChild()), net);

        this.setNumThreads(4);
    }
}