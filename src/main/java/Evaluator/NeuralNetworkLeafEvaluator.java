package Evaluator;

import game.Game;
import main.collections.ChunkSet;
import main.collections.FastArrayList;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import other.context.Context;
import other.move.Move;

import java.util.ArrayList;

/**
 * Generates a leaf evaluator that uses a NN to evaluate states (DeepLearning4J)
 * Please note, this class is NOT thread-safe!!! Use "MultiNeuralNetworkLeafEvaluator" or
 * "ParallelNeuralNetworkLeafEvaluator"
 */
public class NeuralNetworkLeafEvaluator extends GameStateEvaluator {

    //-------------------------------------------------------------------------

    /**
     * MultiLayerNetwork from DeepLearning4J
     */
    protected MultiLayerNetwork net;

    /**
     * INDArray (input for NN) that can be used to convert Ludii's context to NN input
     */
    protected INDArray initialInput;

    /**
     * Total number of squares on the board
     */
    protected int numSquares;

    /**
     * Total number of rows and columns (Squared Board is assumed)
     */
    protected int numRowsCols;

    /**
     * Number of players, assumption of two-player games is made
     */
    protected int numPlayers = 2;

    /**
     * Number of rows/columns with padding (single row is used for all Networks)
     */
    protected int padding = 1;

    //-------------------------------------------------------------------------

    /**
     * Constructor with parameter for game and neural network
     *
     * @param game Ludii's game
     * @param net  DL4J MultiLayerNetwork that needs to be used (can be loaded with the LearningManager).
     *             The network should always predict with respect to player 1.
     */
    public NeuralNetworkLeafEvaluator(Game game, MultiLayerNetwork net) {
        // Declare game information
        this.numSquares = game.board().numSites();
        this.numRowsCols = (int) Math.sqrt(numSquares);

        // Initialise initial input
        // Expected input of NN:          Batch, channel,    board + padding,           board + padding
        this.initialInput = Nd4j.zeros(1, numPlayers, numRowsCols + 2 * padding, numRowsCols + 2 * padding);

        // Save network
        this.net = net;
    }

    /**
     * Evaluates the current context using the NN.
     * The first player (playerID = 1) is assumed to be the maximizing player. The estimated value of the second player
     * (playerID = 2) will be multiplied by -1, since the NN always predicts with respect to playerID 1.
     *
     * @param context          Ludii's context of the current game state
     * @param maximisingPlayer Indicates the playerID of the player to move (either 1 or 2)
     * @return A float value indicating how good the game state is (higher is better)
     */
    public float evaluate(Context context, int maximisingPlayer) {
        if (maximisingPlayer == 1)
            return this.net.output(boardToInput(context), false).getFloat(0);
        else
            return this.net.output(boardToInput(context), false).getFloat(0) * -1;
    }

    /**
     * Evaluates all non-terminal moves of the current context batched using the NN.
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
            return this.net.output(movesToInput(context, nonTerminalMoves), false).toFloatVector();
        else
            return this.net.output(movesToInput(context, nonTerminalMoves), false).mul(-1).toFloatVector();
    }

    /**
     * Converts the game states of the non-terminal moves to an input for DL4J NNs.
     * For each non-terminal move, it generates a channel for each player, and changes the value of a position to 1
     * if a piece of the player is located at that position.
     *
     * @param context          Ludii's context of the current game state
     * @param nonTerminalMoves Index of all moves that are non-terminal (so need to be converted to NN input)
     * @return Multi-channeled matrix of the game state, which can be used for the NN.
     */
    protected INDArray movesToInput(Context context, ArrayList<Integer> nonTerminalMoves) {
        // Create empty input matrix for all non terminal moves
        INDArray boardInput = Nd4j.zeros(nonTerminalMoves.size(), numPlayers,
                numRowsCols + 2 * padding, numRowsCols + 2 * padding);

        // Get all legal moves
        FastArrayList<Move> legalMoves = context.moves(context).moves();

        // For all non-terminal moves
        for (int m = 0; m < nonTerminalMoves.size(); m++) {
            // Generate game state that needs to be converted to NN input
            Context contextCopy = new Context(context);
            contextCopy.game().apply(contextCopy, legalMoves.get(nonTerminalMoves.get(m)));

            // Get chunks of context
            ChunkSet chunkSet = contextCopy.state().containerStates()[0].cloneWhoCell();

            // For both players
            for (int p = 0; p < numPlayers; p++) {
                // Check all positions
                for (int i = 0; i < numSquares; ++i) {
                    // If chunk contains piece of player (value == playerID), change the value to one
                    int value = chunkSet.getChunk(i);
                    if (value == p + 1) {
                        boardInput.putScalar(new int[]{
                                m,
                                p,
                                (int) i / numRowsCols + padding,
                                i % numRowsCols + padding
                        }, 1.0f);
                    }
                }
            }
        }

        return boardInput;
    }

    /**
     * Converts the game board to an input for DL4J NNs.
     * It generates a channel for each player, and changes the value of a position to 1 if a piece of the player
     * is located at that position.
     *
     * @param context Ludii's context of the current game state
     * @return Multi-channeled matrix of the game state, which can be used for the NN.
     */
    public INDArray boardToInput(Context context) {
        // Get chunks of context
        ChunkSet chunkSet = context.state().containerStates()[0].cloneWhoCell();
        INDArray boardInput = initialInput.dup();

        // For both players
        for (int p = 0; p < numPlayers; p++) {
            // Check all positions
            for (int i = 0; i < numSquares; ++i) {
                // If chunk contains piece of player (value == playerID), change the value to one
                int value = chunkSet.getChunk(i);
                if (value == p + 1) {
                    boardInput.putScalar(new int[]{
                            0,
                            p,
                            (int) i / numRowsCols + padding,
                            i % numRowsCols + padding
                    }, 1.0f);
                }
            }
        }

        return boardInput;
    }
}
