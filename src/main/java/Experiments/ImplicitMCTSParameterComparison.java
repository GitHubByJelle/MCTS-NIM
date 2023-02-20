package Experiments;

import game.Game;
import org.nd4j.jita.conf.CudaEnvironment;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import utils.propertyLoader;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Main class which can be used to run tournaments of MCTS bots with different parameters to fine-tune
 * the parameters. It will run a tournament for all given alpha and C combinations.
 */
public class ImplicitMCTSParameterComparison {
    /**
     * Main class which runs a tournament of a single bot pair. It will run a tournament for
     * all given alpha and C combinations. The assumption is made that the first agent is an implicit MCTS
     * bot using a NN. See: Lanctot, M., Winands, M. H., Pepels, T., & Sturtevant, N. R. (2014, August). Monte Carlo
     * tree search with heuristic evaluations using implicit minimax backups. In 2014 IEEE Conference on
     * Computational Intelligence and Games (pp. 1-8). IEEE.
     * @param args All inputs for the tournament. A .proporties file with the following inputs is expected
     *             (see configurations/template.proporties):
     *             String agentClassOne: Name of the class of the first agent
     *             String agentClassTwo: Name of the class of the second agent
     *             String agentOneNN: Path to the NN of the first agent (only required and accepted when agent has "NN" in its name)
     *             String agentTwoNN: Path to the NN of the second agent (only required and accepted when agent has "NN" in its name)
     *             String gameName: Name of the game (as used by Ludii)
     *             int numGames: Total number of games played
     *             float maxSeconds: Maximum number of time per move (in seconds) (-1 means no limit)
     *             int maxIterations: Maximum number of iterations per move (-1 means no limit)
     *             int maxDepth: Maximum searched depth per move (-1 means no limit)
     *             boolean printProgress: Print progress of the number of games played
     *             boolean printResult: Print the results of the tournament
     *             double[] alpha: influence of heuristics of implicit MCTS (multiple values can be given when separated with ",")
     *             double[] C: Exploitation constant (multiple values can be given when separated with ",")
     * @throws IOException
     * @throws ClassNotFoundException
     * @throws NoSuchMethodException
     * @throws InstantiationException
     * @throws IllegalAccessException
     */
    public static void main(String[] args) throws IOException, ClassNotFoundException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        // Check if os is Linux (server), if so use both GPU
        if (System.getProperty("os.name") == "Linux"){
            CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
        }

        // Load properties from file
        Properties props;
        if (args.length == 0) {
            System.out.println("The basic configuration file has been loaded. No input detected.");
            props = propertyLoader.getPropertyValues("configurations\\templateImplicitMCTS.properties");
        } else {
            props = propertyLoader.getPropertyValues(args[0]);
        }

        // Extract all information
        final String agentClassOne = props.getProperty("agentClassOne");
        final String agentClassTwo = props.getProperty("agentClassTwo");
        final String agentOneNN = props.getProperty("agentOneNN");
        final String agentTwoNN = props.getProperty("agentTwoNN");
        final String gameName = props.getProperty("gameName");
        final int numGames = Integer.parseInt(props.getProperty("numGames"));
        final float maxSeconds = Float.parseFloat(props.getProperty("maxSeconds"));
        final int maxIterations = Integer.parseInt(props.getProperty("maxIterations"));
        final int maxDepth = Integer.parseInt(props.getProperty("maxDepth"));
        final boolean printProgress = Boolean.parseBoolean(props.getProperty("printProgress"));
        final boolean printResult = Boolean.parseBoolean(props.getProperty("printResult"));
        final double[] alpha = Stream.of(props.getProperty("alpha").split(","))
                .mapToDouble(Double::parseDouble).toArray();
        final double[] explorationConstant = Stream.of(props.getProperty("C").split(","))
                .mapToDouble(Double::parseDouble).toArray();

        // Assumption of implicit MCTS agent using a NN
        assert agentClassOne.contains("ImplicitMCTSNN");

        // Test the parameters in a tournament
        testParamValues(agentClassOne, agentClassTwo, agentOneNN, alpha, explorationConstant,
                agentTwoNN, gameName, numGames, maxSeconds, maxIterations, maxDepth,
                printProgress, printResult);
    }

    /**
     * Plays games between two bots for both sides for all combinations of alpha and exploration constants
     * and prints results if needed.
     *
     * @param agentClassOne Name of the class of the first agent
     * @param agentClassTwo Name of the class of the second agent
     * @param agentOneNN Path to the NN of the first agent (only required and accepted when agent has "NN" in its name)
     * @param alpha influence of heurisitics of implicit MCTS (multiple values)
     * @param explorationConstant exploration constants of MCTS (multiple values)
     * @param agentTwoNN Path to the NN of the second agent (only required and accepted when agent has "NN" in its name)
     * @param gameName Name of the game (as used by Ludii)
     * @param numGames Total number of games played
     * @param maxSeconds Maximum number of time per move (in seconds) (-1 means no limit)
     * @param maxIterations Maximum number of iterations per move (-1 means no limit)
     * @param maxDepth Maximum searched depth per move (-1 means no limit)
     * @param printProgress Print progress of the number of games played
     * @param printResult Print the results of the tournament
     */
    public static void testParamValues(final String agentClassOne, final String agentClassTwo,
                                      String agentOneNN, double[] alpha, double[] explorationConstant, String agentTwoNN, String gameName,
                                      final int numGames, final float maxSeconds, final int maxIterations,
                                      final int maxDepth, final boolean printProgress, final boolean printResult) {
        // Plays the number of games for both sides
        for (double C : explorationConstant){
            for (double a : alpha){
                testParamMultipleGames(1, agentClassOne, agentClassTwo, agentOneNN, a, C, agentTwoNN, gameName, numGames,
                        maxSeconds, maxIterations, maxDepth, printProgress, printResult);
                testParamMultipleGames(2, agentClassTwo, agentClassOne, agentTwoNN, a, C, agentOneNN, gameName,
                        numGames, maxSeconds, maxIterations, maxDepth, printProgress, printResult);
            }
        }
    }

    /**
     * Plays games between two bots for single side and prints results if needed.
     *
     * @param agentClassOne Name of the class of the first agent
     * @param agentClassTwo Name of the class of the second agent
     * @param agentOneNN Path to the NN of the first agent (only required and accepted when agent has "NN" in its name)
     * @param alpha influence of heuristics of implicit MCTS
     * @param explorationConstant Exploration constant of MCTS
     * @param agentTwoNN Path to the NN of the second agent (only required and accepted when agent has "NN" in its name)
     * @param gameName Name of the game (as used by Ludii)
     * @param numGames Total number of games played
     * @param maxSeconds Maximum number of time per move (in seconds) (-1 means no limit)
     * @param maxIterations Maximum number of iterations per move (-1 means no limit)
     * @param maxDepth Maximum searched depth per move (-1 means no limit)
     * @param printProgress Print progress of the number of games played
     * @param printResult Print the results of the tournament
     */
    public static void testParamMultipleGames(final int side, final String agentClassOne, final String agentClassTwo,
                                         String agentOneNN, final double alpha, final double explorationConstant,
                                              String agentTwoNN, String gameName,
                                         final int numGames, final float maxSeconds, final int maxIterations,
                                         final int maxDepth, boolean printProgress, boolean printResult) {
        AtomicIntegerArray wins = new AtomicIntegerArray(new int[2]);
        AtomicInteger gamesPlayed = new AtomicInteger(0);
        IntStream.range(0, numGames).forEach((i) -> {
            int winner = testParamSingleGame(side, agentClassOne, agentClassTwo, agentOneNN, alpha, explorationConstant,
                    agentTwoNN, gameName, maxSeconds, maxIterations, maxDepth);

            // Update winner
            if (winner > 0) {
                wins.incrementAndGet(winner - 1);
            }

            // Update number of games played
            gamesPlayed.incrementAndGet();

            if (printProgress) {
                System.out.println(gamesPlayed.intValue() + "/" + numGames + " games are finished.");
            }
        });

        // Determine win percentages
        double[] win_percentage = new double[2];
        for (int i = 0; i < wins.length(); i++) {
            win_percentage[i] = (double) wins.get(i) / numGames;
        }
        if (printResult) {
            System.out.println((agentClassOne + (side == 1 ? (" (" + alpha + ", " + explorationConstant + ")") : "" ))
                    + " vs " +
                    (agentClassTwo + (side == 2 ? (" (" + alpha + ", " + explorationConstant + ")") : "" ))
                            + ": " + Arrays.toString(win_percentage));
        }
    }

    /**
     * Plays a single game between to bots.
     *
     * @param side The side of the tested implicit MCTS bot
     * @param agentClassOne Name of the class of the first agent
     * @param agentClassTwo Name of the class of the second agent
     * @param agentOneNN Path to the NN of the first agent (only required and accepted when agent has "NN" in its name)
     * @param alpha influence of heuristics of implicit MCTS
     * @param explorationConstant Exploration constant of MCTS
     * @param agentTwoNN Path to the NN of the second agent (only required and accepted when agent has "NN" in its name)
     * @param gameName Name of the game (as used by Ludii)
     * @param maxSeconds Maximum number of time per move (in seconds) (-1 means no limit)
     * @param maxIterations Maximum number of iterations per move (-1 means no limit)
     * @param maxDepth Maximum searched depth per move (-1 means no limit)
     * @return playerID of the winner
     */
    public static int testParamSingleGame(final int side, final String agentClassOne,
                                          final String agentClassTwo, String agentOneNN, final double alpha,
                                          final double explorationConstant, String agentTwoNN,
                                          String gameName, final float maxSeconds,final int maxIterations,
                                          final int maxDepth) {
        // Set up game & create game environment
        Game game = GameLoader.loadGameFromName(gameName + ".lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);

        // Initialise agents
        final List<AI> agents = new ArrayList<AI>();
        agents.add(null);    // insert null at index 0, because player indices start at 1
        try {
            // Initialise the correct agent with input
            if (agentOneNN != null && agentClassOne.contains("NN")) {
                if (side == 1 && agentClassOne.contains("Implicit")){
                    agents.add((AI) Class.forName(agentClassOne).getDeclaredConstructor(String.class, float.class, float.class)
                            .newInstance(agentOneNN, (float)alpha, (float)explorationConstant));
                }
                else{
                    agents.add((AI) Class.forName(agentClassOne).getDeclaredConstructor(String.class).newInstance(agentOneNN));
                }
            } else if (agentOneNN != null) {
                throw new RuntimeException(agentClassOne + " doesn't use an NN as input. Remove the input or change the agent.");
            } else {
                agents.add((AI) Class.forName(agentClassOne).getDeclaredConstructor().newInstance());
            }
        } catch (InstantiationException | IllegalAccessException | ClassNotFoundException | InvocationTargetException |
                 NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
        try {
            if (agentTwoNN != null && agentClassTwo.contains("NN")) {
                if (side == 2 && agentClassTwo.contains("Implicit")){
                    agents.add((AI) Class.forName(agentClassTwo).getDeclaredConstructor(String.class, float.class, float.class)
                            .newInstance(agentTwoNN, (float)alpha, (float)explorationConstant));
                } else {
                    agents.add((AI) Class.forName(agentClassTwo).getDeclaredConstructor(String.class).newInstance(agentTwoNN));
                }
            } else if (agentTwoNN != null) {
                throw new RuntimeException(agentClassTwo + " doesn't use an NN as input. Remove the input or change the agent.");
            } else {
                agents.add((AI) Class.forName(agentClassTwo).getDeclaredConstructor().newInstance());
            }
        } catch (InstantiationException | IllegalAccessException | ClassNotFoundException | InvocationTargetException |
                 NoSuchMethodException e) {
            throw new RuntimeException(e);
        }

        // Start the game
        game.start(context);

        // (re)initialise our agents
        for (int p = 1; p < agents.size(); ++p) {
            agents.get(p).initAI(game, p);
        }

        // keep going until the game is over
        AI agent = null;
        int mover = 0;
        while (!context.trial().over()) {
            // figure out which player is to move
            mover = context.state().mover();

            // retrieve mover from list of agents
            agent = agents.get(mover);

            // ask agent to select a move
            final Move move = agent.selectAction
                    (
                            game,
                            new Context(context),
                            maxSeconds,
                            maxIterations,
                            maxDepth
                    );

            // apply the chosen move
            game.apply(context, move);
        }

        // Close AI
        for (int i = 1; i <= 2; i++) {
            agents.get(i).closeAI();
        }

        return context.trial().status().winner();
    }
}
