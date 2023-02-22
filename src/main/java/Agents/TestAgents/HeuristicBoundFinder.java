package Agents;

import Evaluator.GameStateEvaluator;
import Evaluator.HeuristicLeafEvaluator;
import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;
import utils.Value;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Bot which can be used to determine the bounds of a given evaluation function (especially Ludii's evaluation function)
 * by playing game
 */
public class HeuristicBoundFinder extends AI {

    //-------------------------------------------------------------------------

    /**
     * Player ID indicating which player this bot is (1 for player 1, 2 for player 2, etc.)
     */
    protected int player = -1;

    /**
     * GameStateEvaluator to evaluate non-terminal leaf nodes
     */
    private GameStateEvaluator leafEvaluator;

    /**
     * Bounds of evaluation function found
     */
    static double[] bounds = new double[]{0, -Value.INF, Value.INF};

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs
     */
    public HeuristicBoundFinder() {
        this.friendlyName = "Example Random AI";
    }

    /**
     * Selects and returns a random action to play, while keeping track of the bounds of the evaluation function.
     *
     * @param game          Reference to the game we're playing.
     * @param context       Copy of the context containing the current state of the game
     * @param MaxSeconds    Max number of seconds before a move should be selected.
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
            final Game game, final Context context, final double MaxSeconds,
            final int maxIterations, final int maxDepth
    ) {
        // Evaluate current state
        float currentValue = this.leafEvaluator.evaluate(context, 1);

        // Check bounds
        if (currentValue > bounds[1])
            bounds[1] = currentValue;
        if (currentValue < bounds[2])
            bounds[2] = currentValue;

        // Get random move
        FastArrayList<Move> legalMoves = game.moves(context).moves();
        final int r = ThreadLocalRandom.current().nextInt(legalMoves.size());

        return legalMoves.get(r);
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
        this.leafEvaluator = new HeuristicLeafEvaluator(game);
    }
}
