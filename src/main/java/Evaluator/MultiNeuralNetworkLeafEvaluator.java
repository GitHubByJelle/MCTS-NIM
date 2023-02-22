package Evaluator;

import game.Game;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import other.context.Context;

import java.util.ArrayList;

/**
 * Generates a leaf evaluator that generates a seperate NeuralNetworkLeafEvaluator (DeepLearning4J)
 * for each individual thread. The evaluator is thread safe, since all threads will use their own Neural Network.
 * Please note, even though it is faster for some configurations than "ParallelNeuralNetworkLeafEvaluator" (since the
 * threads don't need to wait until a batch is filled), more memory is required, since the NN is stored for each thread.
 */
public class MultiNeuralNetworkLeafEvaluator extends NeuralNetworkLeafEvaluator {

    //-------------------------------------------------------------------------

    /**
     * Number of threads that require their own NN evaluator
     */
    private final int nThreads;

    /**
     * Array of NN evaluators for all threads
     */
    private final NeuralNetworkLeafEvaluator[] evaluators;

    //-------------------------------------------------------------------------

    /**
     * Constructor with the game, neural network and number of threads as input
     *
     * @param game     Ludii's game
     * @param net      DL4J MultiLayerNetwork that needs to be used (can be loaded with the LearningManager).
     *                 The network should always predict with respect to player 1.
     * @param nThreads Number of threads that require their own NN evaluator
     */
    public MultiNeuralNetworkLeafEvaluator(Game game, MultiLayerNetwork net, int nThreads) {
        super(game, net);

        // Add number of threads
        this.nThreads = nThreads;

        // Create NeuralNetworkEvaluator for each thread separately in array
        evaluators = new NeuralNetworkLeafEvaluator[nThreads];
        for (int i = 0; i < nThreads; i++) {
            evaluators[i] = new NeuralNetworkLeafEvaluator(game, net.clone());
        }
    }

    /**
     * Evaluates the current context using the NN belonging to the thread being used.
     * The first player (playerID = 1) is assumed to be the maximizing player. The estimated value of the second player
     * (playerID = 2) will be multiplied by -1, since the NN always predicts with respect to playerID 1.
     *
     * @param context          Ludii's context of the current game state
     * @param maximisingPlayer Indicates the playerID of the player to move (either 1 or 2)
     * @return A float value indicating how good the game state is (higher is better)
     */
    public float evaluate(Context context, int maximisingPlayer) {
        return this.evaluators[(int) (Thread.currentThread().threadId() % this.nThreads)].evaluate(context, maximisingPlayer);
    }

    /**
     * Evaluates all non-terminal moves of the current context batched using the NN belonging to the thread being used.
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
        return this.evaluators[(int) (Thread.currentThread().threadId() % this.nThreads)].
                evaluateMoves(context, nonTerminalMoves, maximisingPlayer);
    }
}
