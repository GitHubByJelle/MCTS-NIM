package Evaluator;

import game.Game;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import other.context.Context;

import java.util.ArrayList;

/**
 * Generates a leaf evaluator that uses parallel inference on a NN to evaluate states (DeepLearning4J)
 * The evaluator is thread safe. It will collect the inputs from multiple threads, until a threshold or
 * time limit is reached, after which all threads receive their individual output.
 * See: <a href="https://github.com/deeplearning4j/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/inference/ParallelInferenceExample.java">...</a>
 */
public class ParallelNeuralNetworkLeafEvaluator extends NeuralNetworkLeafEvaluator {

    //-------------------------------------------------------------------------

    /**
     * ParallelInference (DeepLearning4J) which communicates between multiple threads and the NN
     */
    private ParallelInference pi;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the game, neural network and batch Limit as input
     *
     * @param game       Ludii's game
     * @param net        DL4J MultiLayerNetwork that needs to be used (can be loaded with the LearningManager).
     *                   The network should always predict with respect to player 1.
     * @param batchLimit The limit of a single batch being processed by parallel inference at once
     */
    public ParallelNeuralNetworkLeafEvaluator(Game game, MultiLayerNetwork net, int batchLimit) {
        super(game, net);

        // Declare network in parallel
        this.pi = new ParallelInference.Builder(net)
                // BATCHED mode is kind of optimization: if number of incoming requests is too high - PI will be batching individual queries into single batch. If number of requests will be low - queries will be processed without batching
                .inferenceMode(InferenceMode.BATCHED)
                // max size of batch for BATCHED mode. you should set this value with respect to your environment (i.e. gpu memory amounts)
                .batchLimit(batchLimit)
                // set this value to number of available computational devices, either CPUs or GPUs
                .workers(2)
                .build();
    }

    /**
     * Evaluates the current context by sending the current context to the ParallelInference.
     * The first player (playerID = 1) is assumed to be the maximizing player. The estimated value of the second player
     * (playerID = 2) will be multiplied by -1, since the NN always predicts with respect to playerID 1.
     *
     * @param context          Ludii's context of the current game state
     * @param maximisingPlayer Indicates the playerID of the player to move (either 1 or 2)
     * @return A float value indicating how good the game state is (higher is better)
     */
    public float evaluate(Context context, int maximisingPlayer) {
        if (maximisingPlayer == 1)
            return this.net.output(boardToInput(context)).getFloat(0);
        else
            return this.net.output(boardToInput(context)).getFloat(0) * -1;
    }

    /**
     * Evaluates all non-terminal moves of the current context by sending them to the ParallelInference batched.
     * The first player (playerID = 1) is assumed to be the maximizing player. The estimated value of the second player
     * (playerID = 2) will be multiplied by -1, since the NN always predicts with respect to playerID 1.
     *
     * @param context          Ludii's context of the current game state
     * @param nonTerminalMoves Index of all moves that are non-terminal (so need to be converted to NN input)
     * @param maximisingPlayer Indicates the playerID of the player to move (either 1 or 2)
     * @return A float array with a value for each non-terminal move indicating how good
     * the game state is (higher is better)
     */
    public float[] evaluateMoves(Context context, ArrayList<Integer> nonTerminalMoves, int maximisingPlayer) {
        if (maximisingPlayer == 1)
            return this.net.output(movesToInput(context, nonTerminalMoves)).toFloatVector();
        else
            return this.net.output(movesToInput(context, nonTerminalMoves)).mul(-1).toFloatVector();
    }

    /**
     * The ParallelInference requires to be closed to prevent crashes in the code
     * This method needs to be called in the AI agent class in the "CloseAI" method when being used.
     */
    public void close() {
        this.pi.shutdown();
    }
}
