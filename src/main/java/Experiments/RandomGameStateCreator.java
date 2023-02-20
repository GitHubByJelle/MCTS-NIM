package Experiments;

import Agents.RandomAI;
import Evaluator.NeuralNetworkLeafEvaluator;
import Training.LearningManager;
import game.Game;
import org.nd4j.linalg.api.ndarray.INDArray;
import other.AI;
import other.GameLoader;
import other.RankUtils;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import utils.Enums;

import java.util.*;

/**
 * Generates games positions by playing games with random bots
 */
public class RandomGameStateCreator {
    /**
     * Generates Neural Network input of games position by playing with random for a specific amount
     * of plies.
     *
     * @param numPlies number of plies to play
     * @return Neural network input (DeepLearning4J) of game position
     */
    public static INDArray createBoard(int numPlies){
        // Create game environment
        Game game = GameLoader.loadGameFromName("Breakthrough.lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);

        // Initialise agents
        final List<AI> agents = new ArrayList<AI>();
        agents.add(null);    // insert null at index 0, because player indices start at 1
        agents.add(new RandomAI());
        agents.add(new RandomAI());

        // Play games
        // (re)start our game
        game.start(context);

        // (re)initialise our agents
        for (int p = 1; p < agents.size(); ++p) {
            agents.get(p).initAI(game, p);
        }

        // keep going until the game is over
        for (int i = 0; i < numPlies; i++) {
            // figure out which player is to move
            final int mover = context.state().mover();

            // retrieve mover from list of agents
            final AI agent = agents.get(mover);

            // ask agent to select a move
            // we'll give them a search time limit of 0.2 seconds per decision
            // IMPORTANT: pass a copy of the context, not the context object directly
            final Move move = agent.selectAction
                    (
                            game,
                            new Context(context),
                            -1,
                            -1,
                            -1
                    );

            // apply the chosen move
            game.apply(context, move);
        }

        NeuralNetworkLeafEvaluator NNLE = new NeuralNetworkLeafEvaluator(game,
                LearningManager.createNetwork(game, Enums.NetworkType.Cohen, 0.001f));

        return NNLE.boardToInput(context);
    }

    /**
     * Generates Ludii's context of games position by playing with random for a specific amount
     * of plies.
     *
     * @param numPlies number of plies to play
     * @return Ludii's context of game position
     */
    public static Context createContext(int numPlies){
        // Create game environment
        Game game = GameLoader.loadGameFromName("Breakthrough.lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);

        // Initialise agents
        final List<AI> agents = new ArrayList<AI>();
        agents.add(null);    // insert null at index 0, because player indices start at 1
        agents.add(new RandomAI());
        agents.add(new RandomAI());

        // Play games
        // (re)start our game
        game.start(context);

        // (re)initialise our agents
        for (int p = 1; p < agents.size(); ++p) {
            agents.get(p).initAI(game, p);
        }

        // keep going until the game is over
        for (int i = 0; i < numPlies; i++) {
            // figure out which player is to move
            final int mover = context.state().mover();

            // retrieve mover from list of agents
            final AI agent = agents.get(mover);

            // ask agent to select a move
            // we'll give them a search time limit of 0.2 seconds per decision
            // IMPORTANT: pass a copy of the context, not the context object directly
            final Move move = agent.selectAction
                    (
                            game,
                            new Context(context),
                            -1,
                            -1,
                            -1
                    );

            // apply the chosen move
            game.apply(context, move);
        }

        return context;
    }

    /**
     * Generates Neural Network input of games position by playing with random until a
     * terminal game position is reached.
     *
     * @return Neural network input (DeepLearning4J) of terminal game position
     */
    public static INDArray createFinalBoard(){
        // Create game environment
        Game game = GameLoader.loadGameFromName("Breakthrough.lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);

        // Initialise agents
        final List<AI> agents = new ArrayList<AI>();
        agents.add(null);    // insert null at index 0, because player indices start at 1
        agents.add(new RandomAI());
        agents.add(new RandomAI());

        // Play games
        // (re)start our game
        game.start(context);

        // (re)initialise our agents
        for (int p = 1; p < agents.size(); ++p) {
            agents.get(p).initAI(game, p);
        }

        // keep going until the game is over
        while (!context.trial().over()) {
            // figure out which player is to move
            final int mover = context.state().mover();

            // retrieve mover from list of agents
            final AI agent = agents.get(mover);

            // ask agent to select a move
            // we'll give them a search time limit of 0.2 seconds per decision
            // IMPORTANT: pass a copy of the context, not the context object directly
            final Move move = agent.selectAction
                    (
                            game,
                            new Context(context),
                            -1,
                            -1,
                            -1
                    );

            // apply the chosen move
            game.apply(context, move);
        }

        NeuralNetworkLeafEvaluator NNLE = new NeuralNetworkLeafEvaluator(game,
                LearningManager.createNetwork(game, Enums.NetworkType.Cohen, 0.001f));

        System.out.println(RankUtils.utilities(context)[1]);

        return NNLE.boardToInput(context);
    }

    /**
     * Generates Ludii's context of games position by playing with given bots until a
     * terminal game position has been reached.
     *
     * @param agents List of the agents to play the game
     * @return Ludii's context of the terminal game position
     */
    public static Context createFinalContext(final List<AI> agents){
        // Create game environment
        Game game = GameLoader.loadGameFromName("Breakthrough.lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);

        // Play games
        // (re)start our game
        game.start(context);

        // (re)initialise our agents
        for (int p = 1; p < agents.size(); ++p) {
            agents.get(p).initAI(game, p);
        }

        // keep going until the game is over
        while (!context.trial().over()) {
            // figure out which player is to move
            final int mover = context.state().mover();

            // retrieve mover from list of agents
            final AI agent = agents.get(mover);

            // ask agent to select a move
            // we'll give them a search time limit of 0.2 seconds per decision
            // IMPORTANT: pass a copy of the context, not the context object directly
            final Move move = agent.selectAction
                    (
                            game,
                            new Context(context),
                            1,
                            3,
                            -1
                    );

            // apply the chosen move
            game.apply(context, move);
        }

        return context;
    }

    /**
     * Generates Ludii's context of games position by playing with given bots with a specific amount
     * of plies before a terminal game position has been reached.
     *
     * @param agents List of the agents to play the game
     * @param pliesBeforeWin Number of plies before terminal game position
     * @return Ludii's context of the game position
     */
    public static Context createPreFinalContext(List<AI> agents, final int pliesBeforeWin){
        // Play a single game with the given agents
        Context context = RandomGameStateCreator.createFinalContext(agents);

        System.out.printf("Player %d wins!\n", (RankUtils.utilities(context)[1] == 1 ? 1 : 2));

        // Extract all made moves
        List<Move> playedMoves = context.trial().generateRealMovesList();
        int numLegalMoves = playedMoves.size();

        // Create game environment
        Game game = context.game();
        Trial trial = new Trial(game);
        context = new Context(game, trial);

        // Play games
        // (re)start our game
        game.start(context);

        // Keep going until limit of moves is reached
        for (int i = 0; i < numLegalMoves - pliesBeforeWin; i++) {
            // apply the chosen move
            game.apply(context, playedMoves.get(i));
        }

        // Return context that is "k" plies before the end of the game
        return context;
    }
}
