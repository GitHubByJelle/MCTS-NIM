package Agents;

import Evaluator.AdditiveDepthTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import Training.LearningManager;
import game.Game;
import utils.Enums.ExplorationPolicy;
import utils.Enums.SelectionPolicy;
import utils.TranspositionTableStamp;
import utils.TranspositionTableStampCompleted;

/** Selects the best move to play based by using the batched Neural Network variant of the implemented completed descent
 * algorithm in combination with the additive depth heuristic proposed in Cohen-Solal, Q. (2020). Learning to play
 * two-player perfect-information games without knowledge. arXiv preprint arXiv:2008.01188. */
public class descentNNDepthCompleted extends descentNNCompleted {

    //-------------------------------------------------------------------------

    /** Path to the neural network to used by default (NN trained with additive depth heuristic) */
    protected String pathName = "NN_models/Network_bSize128_nEp1_nGa1552_2022-11-13-16-51-52.bin";

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (uses epsilon-greedy exploration policy, safest selection policy and the
     * default network).
     */
    public descentNNDepthCompleted() {
        this.friendlyName = "Descent Depth Completed (Cohen-Solal)";
        this.explorationPolicy = ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = SelectionPolicy.SAFEST;
    }

    /**
     * Constructor with the path to the desired neural network as path (uses epsilon-greedy exploration policy and
     * safest selection policy
     *
     * @param pathName Path to the neural network to be used
     */
    public descentNNDepthCompleted(String pathName) {
        this.friendlyName = "Descent Depth Completed (Cohen-Solal)";
        this.explorationPolicy = ExplorationPolicy.EPSILON_GREEDY;
        this.selectionPolicy = SelectionPolicy.SAFEST;

        this.pathName = pathName;
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
        this.TT = new TranspositionTableStampCompleted(numBitsPrimaryCode);
        this.TT.allocate();

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, LearningManager.loadNetwork(pathName, false));
        this.terminalEvaluator = new AdditiveDepthTerminalStateEvaluator(150);
    }
}


