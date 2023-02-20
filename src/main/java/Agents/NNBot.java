package Agents;

import Evaluator.GameStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import game.Game;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import other.AI;
import other.context.Context;
import other.move.Move;
import utils.Enums.DataSelection;
import utils.TranspositionTableLearning;

/**
 * Public abstract class which can be used for implementing search algorithms that need to be used
 * in the descent framework. Please note, this bot is not expected to be used for game-playing.
 */
public class NNBot extends AI {

    //-------------------------------------------------------------------------

    /** Player ID indicating which player this bot is (1 for player 1, 2 for player 2, etc.) */
    protected int player = -1;

    /** GameStateEvaluator used to evaluate non-terminal leaf nodes (should be a neural network) */
    protected NeuralNetworkLeafEvaluator leafEvaluator;

    /** GameStateEvaluator used to evaluate terminal leaf nodes */
    protected GameStateEvaluator terminalEvaluator;

    /** Enum for the data selection strategy used from the descent framework */
    protected DataSelection dataSelection;

    /** Transposition Table used to store all trainings data */
    protected TranspositionTableLearning TTTraining = null;

    /** Multilayer network (DeepLearning4J) used for GameStateEvaluator */
    protected MultiLayerNetwork net;

    /** Constructor without inputs */
    public NNBot() {
        this.friendlyName = "NN base bot";
    }

    /**
     * Selects and returns an action to play based by using the Neural Network. The search algorithm evaluates
     * all children batched (which improves the performance of the NNs).
     * Please note, this action is not expected to be used, since only the first
     * legal moves is returned.
     *
     * @param game Reference to the game we're playing.
     * @param context Copy of the context containing the current state of the game
     * @param MaxSeconds Max number of seconds before a move should be selected.
     * Values less than 0 mean there is no time limit.
     * @param maxIterations Max number of iterations before a move should be selected.
     * Values less than 0 mean there is no iteration limit.
     * @param maxDepth Max search depth before a move should be selected.
     * Values less than 0 mean there is no search depth limit.
     * @return Preferred move.
     */
    @Override
    public Move selectAction
            (
                    final Game game, final Context context, final double MaxSeconds,
                    final int maxIterations, final int maxDepth
            ) {
        return game.moves(context).moves().get(0);
    }

    /**
     * Perform desired initialisation before starting to play a game
     *
     * @param game The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;
    }

    /**
     * Setter for the Transposition Table used for storing the trainings data
     * @param tt Transposition Table used for storing the trainings data
     */
    public void setTTTraining(TranspositionTableLearning tt) {
        this.TTTraining = tt;
    }

    /**
     * Getter for the Transposition Table used for storing the trainings data
     * @return Transposition Table used for storing the trainings data
     */
    public TranspositionTableLearning getTTTraining() {
        return TTTraining;
    }

    /**
     * Setter for the enum for the data selection strategy used from the descent framework
     * @param dataSelection Enum for the data selection strategy used from the descent framework
     */
    public void setDataSelection(DataSelection dataSelection) {
        this.dataSelection = dataSelection;
    }

    /**
     * Adds the final (terminal) state of an actual game to the Transposition Table with trainings data
     * @param terminalContext Ludii's context class representing a final terminal game position of the actual game
     */
    public void addTerminalStateToTT(Context terminalContext){
        INDArray inputNN = this.leafEvaluator.boardToInput(terminalContext);
        float outputScore = this.terminalEvaluator.evaluate(terminalContext, 1);
        long zobrist = terminalContext.state().fullHash(terminalContext);
        this.TTTraining.store(zobrist, outputScore, 999, inputNN.data().asFloat());
    }
}


