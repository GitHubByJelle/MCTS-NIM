package Experiments;

import Agents.MCTS;
import game.Game;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

/**
 * Main class which runs multiple MCTS bots with different number of threads for a specific amount of times on
 * the starting position of a specified game. To see the iterations, the bots must print them itself.
 */
public class MCTSThreadIterationTest {
    /**
     * Main class which runs multiple MCTS bots with different number of threads with for a specific amount of
     * times on the starting position of a specified game.
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
        // Initialise used MCTS bots (Please note, they should use the IterationWrapper)
//        String[] bots = new String[]{
//                "Agents.TestAgent.ImplicitMCTSNNNP",
//                "Agents.TestAgent.ImplicitMCTSNNNP3",
//                "Agents.TestAgent.ImplicitMCTSNNFET",
//                "Agents.TestAgent.ImplicitMCTSNNFET2",
//                "Agents.ImplicitMCTSNNNPTT",
//                "Agents.TestAgent.ImplicitMCTSNNP",
//                "Agents.TestAgent.ImplicitMCTSNNP2"
//        };

//        String[] bots = new String[]{
//                "Agents.TestAgent.ImplicitMCTSNNNP8",
//                "Agents.TestAgent.ImplicitMCTSNNNP10",
//        };

        String[] bots = new String[]{
                "Agents.ImplicitMCTSNNNP46",
                "Agents.TestAgent.MCTS_ProgressiveBias_MAST",
                "Agents.ImplicitMCTSNNMSP3"
        };

        // Define a list with the number of threads
        int[] numThreads = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

        // Run all MCTS bots N times on the initial board position of the given game for all number of threads
        runThreadsAndBotsNTimes(101, "Breakthrough", bots, numThreads);
    }

    /**
     * Run all MCTS bots N times on the initial board position of the given game for all number of threads
     *
     * @param N          Number of times to search
     * @param gameName   Name of the game
     * @param bots       Array of strings of the used bots
     * @param numThreads Array with number of threads that should be run
     * @throws ClassNotFoundException
     * @throws InvocationTargetException
     * @throws NoSuchMethodException
     * @throws InstantiationException
     * @throws IllegalAccessException
     */
    private static void runThreadsAndBotsNTimes(int N, String gameName, String[] bots, int[] numThreads) throws ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        // Setup game
        Game game = GameLoader.loadGameFromName(gameName + ".lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);
        game.start(context);

        // For all MCTS bots and threads, run the bot N times on the initial game position
        for (String bot : bots) {
            for (int numThread : numThreads) {
                System.out.println(bot + " (" + numThread + "):");

                runThreadAndBotNTimes(N, context, bot, numThread);

                System.out.println();
            }
        }
    }

    /**
     * Run MCTS bot N times on the given game position with the specified number of threads
     *
     * @param N         Number of times to search
     * @param context   Given game position
     * @param bot       String of used MCTS bot
     * @param numThread number of threads used by the MCTS bot
     * @throws ClassNotFoundException
     * @throws NoSuchMethodException
     * @throws InvocationTargetException
     * @throws InstantiationException
     * @throws IllegalAccessException
     */
    private static void runThreadAndBotNTimes(int N, Context context, String bot, int numThread) throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        // Get mover of game position
        int mover = context.state().mover();

        // Initialise agent
        MCTS agent;
        for (int i = 0; i < N; i++) {
            if (bot.contains("NN")) {
                agent = (MCTS) Class.forName(bot)
                        .getDeclaredConstructor(String.class).newInstance("NN_models/Network_bSize128_nEp1_nGa1563_2022-11-12-04-50-34.bin");
            } else {
                agent = (MCTS) Class.forName(bot)
                        .getDeclaredConstructor().newInstance();
            }

            agent.setNumThreads(numThread);
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