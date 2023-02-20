package Agents;

import Evaluator.AdditiveDepthTerminalStateEvaluator;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import game.Game;
import main.collections.FVector;
import main.collections.FastArrayList;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import other.context.Context;
import other.move.Move;
import utils.Enums.ExplorationPolicy;
import utils.Enums.SelectionPolicy;
import utils.TranspositionTableLearning;
import utils.TranspositionTableStamp;
import utils.data_structures.ScoredMove;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/** Selects the best move to play based by using the batched Neural Network variant of the implemented descent
 * algorithm in combination with the additive depth heuristic proposed in Cohen-Solal, Q. (2020). Learning to play
 * two-player perfect-information games without knowledge. arXiv preprint arXiv:2008.01188. It is implemented
 * to additionally store the trainings data found after searching. This allows the search algorithm to be used in the
 * descent framework
 */
public class descentNNDepthTraining extends descentNNTraining {

    /**
     * Constructor with MultiLayerNetwork (DeepLearning4J) as input (uses epsilon-greedy exploration policy and
     * safest selection policy).
     */
    public descentNNDepthTraining(MultiLayerNetwork net) {
        this.friendlyName = "Descent Training";
        this.explorationPolicy = ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = SelectionPolicy.BEST;

        this.net = net;
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Set the playerID, initialise a new Transposition Table and initialise both GameStateEvaluators
     *
     * @param game The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;
        this.TT = new TranspositionTableStamp(numBitsPrimaryCode);
        this.TT.allocate();

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, this.net);
        this.terminalEvaluator = new AdditiveDepthTerminalStateEvaluator(150);
    }
}


