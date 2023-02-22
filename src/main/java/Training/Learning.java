package Training;

import Agents.TestAgent.MCTSTraining;
import Agents.NNBot;
import game.Game;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.jita.conf.CudaEnvironment;
import other.AI;
import other.GameLoader;
import other.RankUtils;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import utils.Enums;
import utils.Enums.NetworkType;
import utils.TranspositionTableLearning;
import utils.propertyLoader;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Properties;

/**
 * Main class which can be used to train the Neural Network.
 */
public class Learning {
    /**
     * Main class to pretrain a Neural network based on the configuration in the .properties file
     *
     * @param args All inputs for pretraining. A .proporties file with the following inputs is expected
     *             (see configurations/templateTraining.proporties):
     *             String agentClass: Name of the class of the agent that is used as search algorithm
     *             String startingNetwork: Path to an existing Neural Network (optional, if not given a new network is created)
     *             String gameName: Name of the game (as used by Ludii)
     *             int numGamesPerIter: Number of games before updating Neural Network (with experience replay it should be updated every game)
     *             float maxSeconds: Maximum number of time per move (in seconds)
     *             int maxIterations: Maximum number of iterations per move (-1 means no limit)
     *             int maxDepth: Maximum searched depth per move (-1 means no limit)
     *             int batchSize: The number of pairs in a single batch during training
     *             int nEpochs: The number of epochs on the created data
     *             int maxSizeExperienceReplay: Maximum number of games in experience replay
     *             float samplingRateExperienceReplay: The rate of the total pairs that are sampled from experience replay
     *             int minutesTraining: The time the descent framework will take to continue training the NN (in minutes)
     *             int numBitsPrimaryCode: The number of bits of the primary code of the Transposition Table that keeps track of the trainings data
     *             float learningRate: The learning rate during training
     *             String trainDataHistoryPath: The path to the experience memory of previous search (can be used to continue training)
     * @throws IOException
     */
    public static void main(String[] args) throws IOException, ClassNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        // Check if os is Linux (server), if so use both GPU
        if (System.getProperty("os.name") == "Linux") {
            CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
        }

        // Load properties from file (or use template)
        Properties props;
        if (args.length == 0) {
            System.out.println("The basic configuration file has been loaded. No input detected.");
            props = propertyLoader.getPropertyValues("configurations\\templateTraining.properties");
        } else {
            props = propertyLoader.getPropertyValues(args[0]);
        }

        // Extract all information
        final String agentClass = props.getProperty("agentClass");
        final String startingNetwork = props.getProperty("startingNetwork");
        final String gameName = props.getProperty("gameName");
        final int numGamesPerIter = Integer.parseInt(props.getProperty("numGamesPerIter"));
        final float maxSeconds = Float.parseFloat(props.getProperty("maxSeconds"));
        final int maxIterations = Integer.parseInt(props.getProperty("maxIterations"));
        final int maxDepth = Integer.parseInt(props.getProperty("maxDepth"));
        final int batchSize = Integer.parseInt(props.getProperty("batchSize"));
        final int nEpochs = Integer.parseInt(props.getProperty("nEpochs"));
        final int maxSizeExperienceReplay = Integer.parseInt(props.getProperty("maxSizeExperienceReplay"));
        final float samplingRateExperienceReplay = Float.parseFloat(props.getProperty("samplingRateExperienceReplay"));
        final int minutesTraining = Integer.parseInt(props.getProperty("minutesTraining"));
        final int numBitsPrimaryCode = Integer.parseInt(props.getProperty("numBitsPrimaryCode"));
        final float learningRate = Float.parseFloat(props.getProperty("learningRate"));
        final String trainDataHistoryPath = props.getProperty("trainDataHistoryPath");

        // Initialize Learning manager with the correct Data Selection
        Game game = GameLoader.loadGameFromName(gameName + ".lud");
        final int boardSize = (int) Math.sqrt(game.board().numSites());
        Enums.DataSelection dataSelection = Enums.DataSelection.TREE;
        LearningManager learningManager = new LearningManager(dataSelection, nEpochs, batchSize,
                maxSizeExperienceReplay, samplingRateExperienceReplay);

        // Initialize neural network
        MultiLayerNetwork net;
        // Initialize new one
        if (startingNetwork == null) {
            net = LearningManager.createNetwork(game, NetworkType.Cohen, learningRate);

            System.out.println("Created new network:");
            System.out.println(net.summary());
        }
        // Reload given model
        else {
            net = LearningManager.loadNetwork(startingNetwork, true);

            System.out.println("Loaded network from " + startingNetwork + ":");
            System.out.println(net.summary());
        }

        // Set train data history (if given)
        if (trainDataHistoryPath != null) {
            learningManager.loadTrainData(trainDataHistoryPath);
        }

        // Calculate stop time in milliseconds
        long stopTimeTraining = System.currentTimeMillis() + minutesTraining * 60 * 1000;

        // When there is time left
        int gameIndex = 0;
        while (System.currentTimeMillis() < stopTimeTraining) {
            // Set up game
            // Create game environment
            game = GameLoader.loadGameFromName(gameName + ".lud");
            Trial trial = new Trial(game);
            Context context = new Context(game, trial);

            // Initialise agents
            final List<AI> agents = new ArrayList<AI>();
            agents.add(null);    // insert null at index 0, because player indices start at 1
            for (int j = 0; j < 2; j++) {
                try {
                    agents.add((AI) Class.forName(agentClass)
                            .getDeclaredConstructor(MultiLayerNetwork.class)
                            .newInstance(net));
                } catch (InstantiationException | IllegalAccessException | InvocationTargetException |
                         NoSuchMethodException | ClassNotFoundException e) {
                    throw new RuntimeException(e);
                }
            }

            // For all games per iterations
            for (int numGame = 0; numGame < numGamesPerIter; numGame++) {
                // (re)start our game
                game.start(context);

                // (re)initialise our agents
                TranspositionTableLearning TTTraining = new TranspositionTableLearning(numBitsPrimaryCode);
                TTTraining.allocate();
                for (int p = 1; p < agents.size(); ++p) {
                    agents.get(p).initAI(game, p);
                    if (agents.get(p) instanceof NNBot) {
                        ((NNBot) agents.get(p)).setTTTraining(TTTraining);
                        ((NNBot) agents.get(p)).setDataSelection(dataSelection);
                    } else if (agents.get(p) instanceof MCTSTraining) {
                        ((MCTSTraining) agents.get(p)).setTTTraining(TTTraining);
                        ((MCTSTraining) agents.get(p)).setDataSelection(dataSelection);
                    } else {
                        throw new RuntimeException("Only use bots of NNBot or MCTSTraining for Learning");
                    }
                }

                // keep going until the game is over
                while (!context.trial().over()) {
                    // figure out which player is to move
                    final int mover = context.state().mover();

                    // retrieve mover from list of agents
                    final AI agent = agents.get(mover);

                    // Select move
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

                // Save data
                if (agents.get(1) instanceof NNBot) {
                    learningManager.saveData(TTTraining, context, (NNBot) agents.get(1), gameIndex, numGame, numGamesPerIter,
                            (float) RankUtils.utilities(context)[1], boardSize);
                } else if (agents.get(1) instanceof MCTSTraining) {
                    learningManager.saveData(TTTraining, context, (MCTSTraining) agents.get(1), gameIndex, numGame, numGamesPerIter,
                            (float) RankUtils.utilities(context)[1], boardSize);
                }

                // Close AI
                for (int i = 1; i <= 2; i++) {
                    agents.get(i).closeAI();
                }
            }

            // Train network
            learningManager.updateNeuralNetwork(net, boardSize, gameIndex);

            // Save checkpoint
            learningManager.saveCheckPoint(net);

            gameIndex += 1;
        }

        // Save network
        learningManager.saveFinalModel(net,
                "Network_" +
                        "bSize" + batchSize + "_" +
                        "nEp" + nEpochs + "_" +
                        "nGa" + gameIndex + "_" +
                        new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date()) +
                        ".bin",
                true);
    }
}
