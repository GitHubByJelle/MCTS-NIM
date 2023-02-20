package Evaluator;

import game.Game;
import metadata.ai.Ai;
import metadata.ai.heuristics.Heuristics;
import metadata.ai.heuristics.terms.HeuristicTerm;
import metadata.ai.heuristics.terms.Material;
import metadata.ai.heuristics.terms.MobilitySimple;
import other.context.Context;

/**
 * Generates a leaf evaluator that uses the build-in heuristic evaluation function of Ludii
 */
public class HeuristicLeafEvaluator extends GameStateEvaluator {

    //-------------------------------------------------------------------------

    /** Ludii's heuristic evaluation function */
    Heuristics heuristicFunction;

    //-------------------------------------------------------------------------

    /**
     * Constructor requiring the game as input
     * @param game Ludii's game
     */
    public HeuristicLeafEvaluator(Game game) {
        // Extract the aiMetadata from the game
        Ai aiMetadata = game.metadata().ai();

        // If the metadata is known, copy the heuristic evaluation function fine-tunes by the Ludii-team,
        // Or use Material and Mobility otherwise (also used by the Ludii team)
        if (aiMetadata != null && aiMetadata.heuristics() != null) {
            this.heuristicFunction = Heuristics.copy(aiMetadata.heuristics());
        } else {
            this.heuristicFunction = new Heuristics(new HeuristicTerm[]{
                    new Material(null, 1.0F, null, null),
                    new MobilitySimple(null, 0.001F)});
        }

        // Initialise the heuristic evaluation function
        this.heuristicFunction.init(game);

    }

    /**
     * Evaluates the current context using the Ludii's heuristics.
     * It calculates the difference between the current player, and the opponents.
     *
     * @param context Ludii's context
     * @param maximisingPlayer The maximising player
     * @return
     */
    @Override
    public float evaluate(Context context, int maximisingPlayer) {
        // Declare needed variables
        int numLegalMoves;
        int opp;

        // Compute the score of the current player
        float heuristicScore = this.heuristicFunction.computeValue(context, maximisingPlayer, 0.001F);

        // Determine opponents
        int[] opponents = this.opponents(maximisingPlayer, context.game().players().count());
        int nOpponents = opponents.length;

        // Extract the score from all the opponents
        for (numLegalMoves = 0; numLegalMoves < nOpponents; ++numLegalMoves) {
            opp = opponents[numLegalMoves];
            if (context.active(opp)) {
                heuristicScore -= this.heuristicFunction.computeValue(context, opp, 0.001F);
            } else if (context.winners().contains(opp)) {
                heuristicScore -= 9999999.0F;
            }
        }

        // Return the score
        return heuristicScore;
    }

    /**
     * Generates an array of opponents for the player, based on the number of players
     *
     * @param player playerID of player being evaluated
     * @param numPlayers Number of players in game
     * @return An array of opponents for the player
     */
    public int[] opponents(int player, int numPlayers) {
        // Generate array
        int[] opponents = new int[numPlayers - 1];
        int idx = 0;

        // For all players, if not current player, add opponent to array
        for (int p = 1; p <= numPlayers; ++p) {
            if (p != player) {
                opponents[idx++] = p;
            }
        }

        return opponents;
    }
}
