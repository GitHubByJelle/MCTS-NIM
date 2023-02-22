package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import Training.LearningManager;
import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;
import utils.EvaluatorUtils;

/**
 * AI which uses one-ply search with a NN to select the move to be played
 */
public class OnePlyNN extends AI {

    //-------------------------------------------------------------------------

    /**
     * Player ID indicating which player this bot is (1 for player 1, 2 for player 2, etc.)
     */
    protected int player = -1;

    /**
     * Neural network to evaluate non-terminal leaf nodes
     */
    private NeuralNetworkLeafEvaluator leafEvaluator;

    /**
     * GameStateEvaluator to evaluate terminal leaf nodes
     */
    private ClassicTerminalStateEvaluator terminalEvaluator;

    /**
     * Path to the neural network to used by default
     */
    private String pathName = "NN_models/Network_bSize128_nEp10_nGa456_2022.10.06.22.07.02";

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs
     */
    public OnePlyNN() {
        this.friendlyName = "One-ply NN";
    }

    /**
     * Constructor with the path to the desired neural network as path
     *
     * @param pathNameNN Path to the neural network to be used
     */
    public OnePlyNN(String pathNameNN) {
        this.friendlyName = "One-ply NN";
        this.pathName = pathNameNN;
    }

    /**
     * Selects and returns an action to play based on a one-ply search using a NN.
     *
     * @param game          Reference to the game we're playing.
     * @param context       Copy of the context containing the current state of the game
     * @param maxSeconds    Max number of seconds before a move should be selected.
     *                      Values less than 0 mean there is no time limit.
     * @param maxIterations Max number of iterations before a move should be selected.
     *                      Values less than 0 mean there is no iteration limit.
     * @param maxDepth      Max search depth before a move should be selected.
     *                      Values less than 0 mean there is no search depth limit.
     * @return Preferred move.
     */
    @Override
    public Move selectAction
    (
            final Game game, final Context context, final double maxSeconds,
            final int maxIterations, final int maxDepth
    ) {
        // Extract the player to move
        int maximisingPlayer = context.state().playerToAgent(context.state().mover());

        // Extract all legal moves
        FastArrayList<Move> legalMoves = game.moves(context).moves();

        // Evaluate all children
        double[] values = EvaluatorUtils.EvaluateChildrenBatched(context, legalMoves.toArray(new Move[0]),
                maximisingPlayer, this.leafEvaluator, this.terminalEvaluator);

        // Select the best move
        Move bestMove = legalMoves.get(0);
        double bestValue = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > bestValue) {
                bestMove = legalMoves.get(i);
            }
        }

        return bestMove;
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Set the playerID and initialise both GameStateEvaluators
     *
     * @param game     The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;

        this.leafEvaluator = new NeuralNetworkLeafEvaluator(game, LearningManager.loadNetwork(pathName, false));
        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
    }
}
