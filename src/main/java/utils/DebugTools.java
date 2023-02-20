package utils;

import game.Game;
import main.collections.ChunkSet;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

import java.io.*;
import java.util.List;

/**
 * Class containing tools which can be used to debug bots more efficient.
 */
public class DebugTools {
    /**
     * Converts Ludii's context to a string. A squared boards is assumed.
     *
     * @param context Ludii's context
     * @return A string representing the game position
     */
    public static String contextToString(Context context){
        // Initialise all needed fields
        String output = "";
        int numPositions = context.game().board().numSites();
        int numRowCols = (int)Math.sqrt(numPositions); // Squared board is assumed, with two players

        // Load all pieces
        ChunkSet chunkSet = context.state().containerStates()[0].cloneWhoCell();

        // For all rows and colums, add character for correct piece type
        // Use reversed row order when selecting piece, since origin is at left bottom (last row of the string)
        for (int r = 0; r < numRowCols ; r++) {
            // Add index
            output += String.format("%2d | ", (numRowCols - r - 1) * numRowCols);
            for (int c = 0; c < numRowCols; c++) {
                // Select character based on piece id
                int value = chunkSet.getChunk((numRowCols - r - 1) * numRowCols + c);
                if (value == 0){
                    output += " . ";
                } else if (value == 1){
                    output += " w ";
                } else if (value == 2){
                    output += " b ";
                }
            }
            // Start new row
            output += "\n";
        }

        return output;
    }

    /**
     * Load existing context by loading the moves made and playing them
     *
     * @param pathPlayedGames Path to file that contains all made moves during the game
     * @param gameName Name of the game
     * @return Game position after playing the loaded moves
     */
    public static Context loadExistingContext(String pathPlayedGames, String gameName){
        // Load all made moves
        List<Move> playedMoves;
        try (ObjectInputStream movesInput = new ObjectInputStream(new FileInputStream(pathPlayedGames))) {
            Object readObject = movesInput.readObject();

            playedMoves = (List<Move>)readObject;
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }

        int numLegalMoves = playedMoves.size();

        // Create game environment
        Game game = GameLoader.loadGameFromName(gameName + ".lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);

        // Play games
        // (re)start our game
        game.start(context);

        // Keep going until limit of moves is reached
        for (int i = 0; i < numLegalMoves; i++) {
            // apply the chosen move
            game.apply(context, playedMoves.get(i));
        }

        // Return context that is "k" plies before the end of the game
        return context;
    }

    /**
     * Save the "context" of a given game position by saving all played moves
     *
     * @param context Ludii's context
     * @param name Name of the save file for the played moves
     */
    public static void saveContext(Context context, String name){
        // Uncomment to save context (can be used for debugging purposes)
        try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
                new FileOutputStream(name + ".obj"))) {
            trainExamplesOutput.writeObject(context.trial().generateRealMovesList());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
