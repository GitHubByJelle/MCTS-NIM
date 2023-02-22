package Experiments;

import game.Game;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

/**
 * Main class which runs multiple bots for a specific amount of times on the starting position of a specified game.
 * To see the iterations, the bots must print them itself.
 */
public class IterationTest {
    /**
     * Main class which runs multiple bots for a specific amount of times on the starting position of a specified game.
     *
     * @param args No input
     * @throws IOException
     * @throws ClassNotFoundException
     * @throws InvocationTargetException
     * @throws NoSuchMethodException
     * @throws InstantiationException
     * @throws IllegalAccessException
     */
    public static void main(String[] args) throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        // Initialise used bots (Please note, they should print the number of iterations)
//        String[] bots = new String[]{
//                "Agents.ImplicitMCTSNNNP",
//                "Agents.ImplicitMCTSNNNP3",
//                "Agents.ImplicitMCTSNNFET",
//                "Agents.ImplicitMCTSNNFET2",
//                "Agents.ImplicitMCTSNNNPTT",
//                "Agents.ImplicitMCTSNNP",
//                "Agents.ImplicitMCTSNNP2",
//                "Agents.UBFMNNCompleted"
//        };

        String[] bots = new String[]{
                "Agents.AlphaBetaSearchHF",
                "Agents.MCTSTest.MCTS_ProgressiveBias_MAST"
        };

        // Run all bots N times on the initial board position of the given game
        runBotsNTimes(101, "Breakthrough", bots);
    }

    /**
     * Run all bots N times on the initial board position of the given game.
     *
     * @param N        Number of times to search
     * @param gameName Name of the game
     * @param bots     Array of strings of the used bots
     * @throws ClassNotFoundException
     * @throws InvocationTargetException
     * @throws NoSuchMethodException
     * @throws InstantiationException
     * @throws IllegalAccessException
     */
    private static void runBotsNTimes(int N, String gameName, String[] bots) throws ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        // Setup game
        Game game = GameLoader.loadGameFromName(gameName + ".lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);
        game.start(context);

        // For all bots, run the bot N times on the initial game position
        for (String bot : bots) {
            System.out.println(bot + ":");

            runBotNTimes(N, context, bot);

            System.out.println();
        }
    }


    /**
     * Run bot N times on the given game position
     *
     * @param N       Number of times to search
     * @param context Given game position
     * @param bot     String of used bot
     * @throws ClassNotFoundException
     * @throws NoSuchMethodException
     * @throws InvocationTargetException
     * @throws InstantiationException
     * @throws IllegalAccessException
     */
    private static void runBotNTimes(int N, Context context, String bot) throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        // Get mover of game position
        int mover = context.state().mover();

        // Initialise agent
        for (int i = 0; i < N; i++) {
//            AI agent = (AI) Class.forName(bot)
//                    .getDeclaredConstructor(String.class).newInstance("NN_models/Network_bSize128_nEp1_nGa1563_2022-11-12-04-50-34.bin");
            AI agent = (AI) Class.forName(bot)
                    .getDeclaredConstructor().newInstance();
            agent.initAI(context.game(), mover);

            // Search (to see iterations this action should print the iterations)
            final Move move = agent.selectAction
                    (
                            context.game(),
                            new Context(context),
                            1,
                            -1,
                            -1
                    );

            agent.closeAI();
        }
    }
}