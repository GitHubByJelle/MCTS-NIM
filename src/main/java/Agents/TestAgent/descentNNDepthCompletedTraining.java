package Agents.TestAgent;

import Agents.descentNNCompletedTraining;
import Evaluator.AdditiveDepthTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import game.Game;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import utils.Enums.ExplorationPolicy;
import utils.Enums.SelectionPolicy;
import utils.TranspositionTableStampCompleted;

/**
 * Selects the best move to play based by using the batched Neural Network variant of the implemented completed descent
 * algorithm in combination with the additive depth heuristic proposed in Cohen-Solal, Q. (2020). Learning to play
 * two-player perfect-information games without knowledge. arXiv preprint arXiv:2008.01188. It is implemented
 * to additionally store the trainings data found after searching. This allows the search algorithm to be used in the
 * descent framework
 */
public class descentNNDepthCompletedTraining extends descentNNCompletedTraining {

    /**
     * Constructor with MultiLayerNetwork (DeepLearning4J) as input (uses epsilon-greedy exploration policy and
     * safest selection policy).
     */
    public descentNNDepthCompletedTraining(MultiLayerNetwork net) {
        this.friendlyName = "Descent Completed Training";
        this.explorationPolicy = ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = SelectionPolicy.BEST;

        this.net = net;
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Set the playerID, initialise a new Transposition Table and initialise both GameStateEvaluators
     *
     * @param game     The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;
        this.TT = new TranspositionTableStampCompleted(numBitsPrimaryCode);
        this.TT.allocate();

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, this.net);
        this.terminalEvaluator = new AdditiveDepthTerminalStateEvaluator(150);
    }
}


