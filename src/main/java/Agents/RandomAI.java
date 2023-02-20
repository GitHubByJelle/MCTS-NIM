package Agents;

import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;
import utils.DebugTools;

import java.util.concurrent.ThreadLocalRandom;

/** Random AI which selects a random index to be played */
public class RandomAI extends AI {

    //-------------------------------------------------------------------------

    /** Player ID indicating which player this bot is (1 for player 1, 2 for player 2, etc.) */
    protected int player = -1;

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs
     */
    public RandomAI() {
        this.friendlyName = "Random AI";
    }

    /**
     * Selects and returns a random action to play.
     *
     * @param game Reference to the game we're playing.
     * @param context Copy of the context containing the current state of the game
     * @param maxSeconds Max number of seconds before a move should be selected.
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
                    final Game game, final Context context, final double maxSeconds,
                    final int maxIterations, final int maxDepth
            ) {
        FastArrayList<Move> legalMoves = game.moves(context).moves();
        final int r = ThreadLocalRandom.current().nextInt(legalMoves.size());
        return legalMoves.get(r);
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
}
