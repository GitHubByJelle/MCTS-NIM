package Evaluator;

import game.Game;
import main.collections.ChunkSet;
import other.context.Context;

/**
 * Generates a leaf evaluator for the game of Breakthrough proposed by Maarten Schadd in:
 * M. P. D. Schadd. Selective Search in Games of Different Complexity. PhD thesis, Maastricht
 * University, Maastricht, The Netherlands, 2011.
 * , but scaled as in:
 * Lanctot, M., Winands, M. H., Pepels, T., & Sturtevant, N. R. (2014, August). Monte Carlo tree search with
 * heuristic evaluations using implicit minimax backups. In 2014 IEEE Conference on Computational Intelligence
 * and Games (pp. 1-8). IEEE.
 */
public class MSLeafEvaluator extends GameStateEvaluator {

    //-------------------------------------------------------------------------

    /**
     * Total number of squares on the board
     */
    protected int numSquares;

    /**
     * Total number of rows and columns (Squared Board is assumed)
     */
    protected int numRowsCols;

    //-------------------------------------------------------------------------

    /**
     * Constructur with the game as input
     *
     * @param game Ludii's game
     */
    public MSLeafEvaluator(Game game) {
        // Declare game information
        this.numSquares = game.board().numSites();
        this.numRowsCols = (int) Math.sqrt(numSquares);
    }

    /**
     * Evaluates the current context using the evaluation function proposed by Maarten Schadd, but implemented
     * as Lanctot.
     * <p>
     * For each player the furthest row is calculated which is worth 2.5 * the number of rows. Each piece is worth 10.
     * The difference between two players is calculated.
     *
     * @param context          Ludii's context
     * @param maximisingPlayer The maximising player
     * @return
     */
    @Override
    public float evaluate(Context context, int maximisingPlayer) {
        // Declare needed variables
        float score = 0;
        int playerOneBestRow = -1;
        int playerTwoBestRow = this.numRowsCols + 1;

        // Extract chunks from context
        ChunkSet chunkSet = context.state().containerStates()[0].cloneWhoCell();

        // For all squares, determine the furthest row for both players and number of pieces
        float value;
        int row;
        for (int i = 0; i < this.numSquares; i++) {
            value = chunkSet.getChunk(i);
            if (value == 1) {
                score += 10;

                row = (int) i / this.numRowsCols;
                if (row > playerOneBestRow) {
                    playerOneBestRow = row;
                }
            } else if (value == 2) {
                score -= 10;

                row = (int) i / this.numRowsCols;
                if (row < playerTwoBestRow) {
                    playerTwoBestRow = row;
                }
            }
        }

        // Add furthest row
        score += (playerOneBestRow - (this.numRowsCols - 1 - playerTwoBestRow)) * 2.5f;

        // If not player 1, minimize
        if (maximisingPlayer != 1) {
            score *= -1;
        }

        return score;
    }
}
