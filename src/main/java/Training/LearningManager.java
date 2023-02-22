package Training;

import Agents.TestAgent.MCTSTraining;
import Agents.NNBot;
import Agents.RandomAI;
import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.GameStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import game.Game;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import utils.Enums.DataSelection;
import utils.Enums.NetworkType;
import utils.TranspositionTableLearning;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

/**
 * Manages everything with respect to training a Neural Network. Based on the descent framework, proposed in:
 * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
 * knowledge. arXiv preprint arXiv:2008.01188.
 */
public class LearningManager {

    //-------------------------------------------------------------------------

    /**
     * Enum for the data selection strategy used from the descent framework
     */
    DataSelection dataSelection;

    /**
     * The number of epochs on the created data
     */
    int nEpochs;

    /**
     * The number of TrainingSamples in a single batch during training
     */
    int batchSize;

    /**
     * The number of players in the game (two players are assumed)
     */
    final static int numPlayers = 2;

    /**
     * Padding added to the input of the NN (padding of 1 is assumed)
     */
    final static int padding = 1;

    /**
     * Maximum number of games in experience replay
     */
    int maxSizeExperienceReplay;

    /**
     * The rate of the total pairs that are sampled from experience replay
     */
    float samplingRateExperienceReplay;

    /**
     * Randomgenerator to generate pseudorandom numbers
     */
    public static Random rng;

    /**
     * File class for the last created Neural Network (can be used to continue training)
     */
    private final File checkpointFile = new File("Network_checkpoint.bin");

    /**
     * Filename of the file that will be created to create checkpoints of the experience memory
     */
    private final String checkpointTrainData = "TrainData_checkpoint.obj";

    /**
     * Hashmap which stores all training data (experience memory)
     */
    private static LinkedHashMap<String, TrainingSample> trainData = new LinkedHashMap<>();

    //-------------------------------------------------------------------------

    /**
     * Constructor with the data selection, number of epochs, batch size, maximum size
     * of the experience replay and sample rate of the experience replay as input.
     *
     * @param dataSelection                Data selection strategy used from the descent framework
     * @param nEpochs                      The number of epochs on the created data
     * @param batchSize                    The number of pairs in a single batch during training
     * @param maxSizeExperienceReplay      Maximum number of games in experience replay
     * @param samplingRateExperienceReplay The rate of the total pairs that are sampled from experience replay
     */
    public LearningManager(DataSelection dataSelection, int nEpochs, int batchSize,
                           int maxSizeExperienceReplay, float samplingRateExperienceReplay) {
        this.dataSelection = dataSelection;
        this.nEpochs = nEpochs;
        this.batchSize = batchSize;
        this.maxSizeExperienceReplay = maxSizeExperienceReplay;
        this.samplingRateExperienceReplay = samplingRateExperienceReplay;

        this.rng = new Random();
    }

    /**
     * Load a Neural Network (DeepLearning4J) based on the given path
     *
     * @param pathName    Path to an existing Neural Network
     * @param loadUpdater Indicates if NN needs to be used for Training (true means yes)
     * @return Neural Network (DeepLearning4J) class
     */
    public static MultiLayerNetwork loadNetwork(String pathName, boolean loadUpdater) {
        // Load model
        try {
            return MultiLayerNetwork.load(new File(pathName), loadUpdater);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Creates a new neural network based on the architecture and game given
     *
     * @param game         Ludii's game class
     * @param networkType  Enum for the Neural Network architecture used
     * @param learningRate The learning rate during training
     * @return Untrained Neural Network (DeepLearning4J)
     */
    public static MultiLayerNetwork createNetwork(Game game, NetworkType networkType, float learningRate) {
        MultiLayerNetwork createdNet;
        switch (networkType) {
            case TicTacToe:
                createdNet = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .weightInit(WeightInit.RELU)
                        .updater(new Adam(learningRate))
                        .l2(0.001)
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(3, 3)
                                .nIn(2) // Num Channels
                                .stride(1, 1)
                                .nOut(64)
                                .activation(Activation.RELU)
                                .build())
                        .layer(1, new ConvolutionLayer.Builder(3, 3)
                                .stride(1, 1)
                                .nOut(64)
                                .activation(Activation.RELU)
                                .build())
                        .layer(2, new DenseLayer.Builder()
                                .nOut(100)
                                .activation(Activation.RELU)
                                .build())
                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .activation(Activation.TANH)
                                .nOut(1).build())
                        .setInputType(InputType.convolutional((int) Math.sqrt(game.board().numSites()) + 2 * padding,
                                (int) Math.sqrt(game.board().numSites()) + 2 * padding, 2))
                        .build()
                );
                createdNet.init();
                createdNet.setListeners(new ScoreIterationListener(1));
//                createdNet.setListeners(new PerformanceListener(1));
                break;
            case Cohen:
                createdNet = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .weightInit(WeightInit.RELU)
                        .updater(new Adam(learningRate))
                        .l2(0.001)
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(3, 3)
                                .nIn(2) // Num Channels
                                .stride(1, 1)
                                .nOut(64)
                                .activation(Activation.RELU)
                                .build())
                        .layer(1, new ConvolutionLayer.Builder(3, 3)
                                .stride(1, 1)
                                .nOut(64)
                                .activation(Activation.RELU)
                                .build())
                        .layer(2, new ConvolutionLayer.Builder(3, 3)
                                .stride(1, 1)
                                .nOut(64)
                                .activation(Activation.RELU)
                                .build())
                        .layer(3, new DenseLayer.Builder()
                                .nOut(100)
                                .activation(Activation.RELU)
                                .build())
                        .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .activation(Activation.TANH)
                                .nOut(1).build())
                        .setInputType(InputType.convolutional((int) Math.sqrt(game.board().numSites()) + 2 * padding,
                                (int) Math.sqrt(game.board().numSites()) + 2 * padding, 2))
                        .build()
                );
                createdNet.init();
                createdNet.setListeners(new ScoreIterationListener(1));
//                createdNet.setListeners(new PerformanceListener(1));
                break;
            default:
                throw new RuntimeException("Unkown network type");
        }

        return createdNet;
    }

    /**
     * Reloads train data from a previous games which can be used to continue training
     *
     * @param pathName The path to the experience memory of the last few searches of previous training
     */
    public void loadTrainData(String pathName) {
        // Loaad object
        try (ObjectInputStream trainExamplesInput = new ObjectInputStream(new FileInputStream(pathName))) {
            Object readObject = trainExamplesInput.readObject();

            // Add to data to learningmanager
            this.trainData = (LinkedHashMap<String, TrainingSample>) readObject;

            // Since the old game indices have been used, the indices needs to be corrected
            this.resetGameIndex(this.trainData);

            // Update user
            System.out.printf("Restored train examples from %s with %d train examples.\n",
                    pathName,
                    this.trainData.size());
        } catch (ClassNotFoundException e) {
            System.out.println("Train examples could not be restored. Continue with empty train examples history.");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Resets the game index of the train data. Required to use when starting new training. It
     * makes sure the correct samples are used in the experience memory.
     * E.g. Samples of games 40, 41 and 42 are changed to -3, -2 and -1.
     *
     * @param linkedHashMap TrainingSamples with corrected index
     */
    private void resetGameIndex(LinkedHashMap<String, TrainingSample> linkedHashMap) {
        // Determine size and current index
        int sizeData = linkedHashMap.size();
        String[] keys = Arrays.copyOf(linkedHashMap.keySet().toArray(), sizeData, String[].class);
        int currentIndex = linkedHashMap.get(keys[sizeData - 1]).index;
        int currentNewIndex = -1;

        // For all sample, reset the index
        for (int i = sizeData - 1; i >= 0; i--) {
            String key = keys[i];
            TrainingSample sample = linkedHashMap.get(key);
            // Reset index
            if (currentIndex == sample.index) {
                sample.index = currentNewIndex;
            }
            // Decrement the current index by 1 if a new index has been found
            else {
                currentIndex = sample.index;
                currentNewIndex--;

                sample.index = currentNewIndex;
            }
        }
    }

    /**
     * Saves the data from the Transposition Table (created by the Neural Network) from last game
     * to the experience memory.
     *
     * @param newData         Transposition Table used by search algorithm during the game
     * @param terminalContext Ludii's context of the terminal game position of the played game
     * @param agent           NNBot class of agent (search algorithm) used during Training
     * @param gameIndex       Index of the current game
     * @param numGame         Number of games from current iteration
     * @param numGamesPerIter Number of games from all iterations
     * @param finalScore      Score of terminal game position (determined by TerminalGameStateEvaluator)
     * @param boardSize       Number of squares on the board (squared board is assumed)
     */
    public void saveData(TranspositionTableLearning newData, Context terminalContext, NNBot agent,
                         int gameIndex, int numGame, int numGamesPerIter, float finalScore,
                         int boardSize) {
        // Add terminal state of game played by bot (not seen during play)
        agent.addTerminalStateToTT(terminalContext);

        // Extract data from TT to a hashmap
        HashMap<String, TrainingSample> gameExamples = getDataFromTT(newData, this.dataSelection, finalScore, gameIndex);

        // Add samples based on symmetry
        gameExamples = createMirroredBoards(gameExamples, boardSize);

        // Add the new samples to the trainingdata
        this.addNewSamples(this.trainData, gameExamples);

        System.out.println("Game " + (numGame + 1) + "/" + numGamesPerIter +
                " finished. Total training samples: " + this.trainData.size() +
                " (" + gameExamples.size() + " new).");
    }

    /**
     * Saves the data from the Transposition Table (created by the Neural Network) from last game
     * to the experience memory.
     *
     * @param newData         Transposition Table used by search algorithm during the game
     * @param terminalContext Ludii's context of the terminal game position of the played game
     * @param agent           MCTS class of agent (search algorithm) used during Training
     * @param gameIndex       Index of the current game
     * @param numGame         Number of games from current iteration
     * @param numGamesPerIter Number of games from all iterations
     * @param finalScore      Score of terminal game position (determined by TerminalGameStateEvaluator)
     * @param boardSize       Number of squares on the board (squared board is assumed)
     */
    public void saveData(TranspositionTableLearning newData, Context terminalContext, MCTSTraining agent,
                         int gameIndex, int numGame, int numGamesPerIter, float finalScore,
                         int boardSize) {
        // Add terminal state of game played by bot (not seen during play)
        agent.addTerminalStateToTT(terminalContext);

        // Extract data from TT to a hashmap
        HashMap<String, TrainingSample> gameExamples = getDataFromTT(newData, this.dataSelection, finalScore, gameIndex);

        // Add samples based on symmetry
        gameExamples = createMirroredBoards(gameExamples, boardSize);

        // Add the new samples to the trainingdata
        this.addNewSamples(this.trainData, gameExamples);

        System.out.println("Game " + (numGame + 1) + "/" + numGamesPerIter +
                " finished. Total training samples: " + this.trainData.size() +
                " (" + gameExamples.size() + " new).");
    }

    /**
     * Saves the data from the Transposition Table (created by the Neural Network) from the games
     * that have been run in parallel to the shared experience memory (synchronized)
     *
     * @param newData         Transposition Table used by search algorithm during the game
     * @param gameIndex       Index of the current game
     * @param numGame         Number of games from current iteration
     * @param numGamesPerIter Number of games from all iterations
     * @param finalScore      Score of terminal game position (determined by TerminalGameStateEvaluator)
     * @param boardSize       Width the board in squares (squared board is assumed)
     */
    public void saveDataParallel(TranspositionTableLearning newData, int gameIndex, int numGame, int numGamesPerIter,
                                 float finalScore, int boardSize) {
        // Extract data from TT to a hashmap
        HashMap<String, TrainingSample> gameExamples = getDataFromTT(newData, this.dataSelection, finalScore, gameIndex);

        // Add samples based on symmetry
        gameExamples = createMirroredBoards(gameExamples, boardSize);

        // Add the new samples to the trainingdata
        synchronized (this.trainData) {
            this.addNewSamples(this.trainData, gameExamples);
        }

        System.out.println("Game " + (numGame + 1) + "/" + numGamesPerIter +
                " finished. Total training samples: " + this.trainData.size() +
                " (" + gameExamples.size() + " new).");
    }

    /**
     * Performs data augmentation on the board position. This implementation is really based on a game like
     * Breakthrough were the board can be flipped vertically. Additionally, the switching the perspectives of the
     * players results starting position that is -1 * with his own starting position can be added.
     *
     * @param originalSamples The TrainingSamples which needed to be mirrored
     * @param boardSize       Width the board in squares (squared board is assumed)
     * @return Mirrored TrainingSamples INCLUDING the original TrainingSamples
     */
    private HashMap<String, TrainingSample> createMirroredBoards(HashMap<String, TrainingSample> originalSamples,
                                                                 int boardSize) {
        // Initialise needed variables
        INDArray mirroredBoard;
        float[] floatMirroredBoard;
        HashMap<String, TrainingSample> mirroredSamples = new HashMap<>();

        for (TrainingSample trainSample : originalSamples.values()) {
            // Create original board
            INDArray originalBoard = Nd4j.create(trainSample.board, 1, numPlayers,
                    boardSize + 2 * padding, boardSize + 2 * padding);

            // Add original board
            mirroredSamples.put(Arrays.toString(trainSample.board),
                    new TrainingSample(trainSample.board, trainSample.value, trainSample.index));

            // Add vertical mirroredboard
            mirroredBoard = mirrorBoardVertically(originalBoard);
            floatMirroredBoard = mirroredBoard.data().asFloat();

            if (!mirroredSamples.containsKey(Arrays.toString(floatMirroredBoard))) {
                mirroredSamples.put(Arrays.toString(floatMirroredBoard),
                        new TrainingSample(floatMirroredBoard, trainSample.value, trainSample.index));
            }

//            // Add for perspective of other player (so value multiplied by -1)
//            mirroredBoard = mirrorPlayerPerspective(originalBoard);
//            floatMirroredBoard = mirroredBoard.data().asFloat();
//            if (!mirroredSamples.containsKey(mirroredBoard)){
//                mirroredSamples.put(Arrays.toString(floatMirroredBoard),
//                        new TrainingSample(floatMirroredBoard, -trainSample.value, trainSample.index));
//            }
//
//            // Add for perspective of other player and vertically mirrored
//            mirroredBoard = mirrorBoardVertically(mirroredBoard);
//            floatMirroredBoard = mirroredBoard.data().asFloat();
//            if (!mirroredSamples.containsKey(mirroredBoard)){
//                mirroredSamples.put(Arrays.toString(floatMirroredBoard),
//                        new TrainingSample(floatMirroredBoard, -trainSample.value, trainSample.index));
//            }
        }

        return mirroredSamples;
    }

    /**
     * Merges to hashmaps
     *
     * @param linkedHashMap HashMap to which the samples need to be added
     * @param newSamples    Data wished to add to the other HashMap
     */
    private void addNewSamples(LinkedHashMap<String, TrainingSample> linkedHashMap,
                               HashMap<String, TrainingSample> newSamples) {
        // Add new samples to the trainData, if board (key) is already present, save the newest
        for (String newBoardKey : newSamples.keySet()) {
            linkedHashMap.remove(newBoardKey);
            linkedHashMap.put(newBoardKey, newSamples.get(newBoardKey));
        }
    }

    /**
     * Converts the data from the Transposition Table to hashmap. The assumption is made that the
     * Transposition Tables only contains the Samples that belong to the data selection.
     *
     * @param TT            Transposition Table used by search algorithm during the game
     * @param dataSelection Data selection strategy used from the descent framework
     * @param finalScore    Score of terminal game position (determined by TerminalGameStateEvaluator)
     * @param gameIndex     Index of the current game
     * @return Hashmap of TrainingSamples of current game
     */
    private static HashMap<String, TrainingSample> getDataFromTT(TranspositionTableLearning TT, DataSelection dataSelection,
                                                                 float finalScore, int gameIndex) {

        // Initialise needed variables
        HashMap<String, TrainingSample> gameSamples = new HashMap<>();
        TranspositionTableLearning.learningTTData data;
        TranspositionTableLearning.learningTTEntry[] table = TT.getTable();
        TranspositionTableLearning.learningTTEntry entry;

        // Based on data selection strategy, extract the correct samples from the Transposition Table
        switch (dataSelection) {
            case ROOT:
            case TREE:
                // For all entries in the tree
                for (int i = 0; i < table.length; i++) {
                    entry = table[i];
                    // If it isn't empty
                    if (entry != null) {
                        // Check all TTData on that entry
                        for (int j = 0; j < entry.data.size(); j++) {
                            data = entry.data.get(j);
                            gameSamples.put(Arrays.toString(data.inputNN),
                                    new TrainingSample(data.inputNN, data.value, gameIndex));
                        }
                    }
                }
                break;
            case TERMINAL:
                // For all entries in the tree
                for (int i = 0; i < table.length; i++) {
                    entry = table[i];
                    // If it isn't empty
                    if (entry != null) {
                        // Check all TTData on that entry
                        for (int j = 0; j < entry.data.size(); j++) {
                            data = entry.data.get(j);
                            gameSamples.put(Arrays.toString(data.inputNN),
                                    new TrainingSample(data.inputNN, finalScore, gameIndex));
                        }
                    }
                }
                break;
            // If non-leaf or terminal, add
            default:
                throw new RuntimeException("Unkown data selection");
        }

        return gameSamples;
    }

    /**
     * Updates the weights of the Neural Network with a sample from the experience memory
     *
     * @param net       Neural Network class
     * @param boardSize Width of the board (squared board is assumed)
     * @param gameIndex Index of the current game
     */
    public void updateNeuralNetwork(MultiLayerNetwork net, int boardSize, int gameIndex) {
        // Process data from experience memory to a sampled input for the NN
        List<DataSet> dataset = this.processData(boardSize, gameIndex);

        // Fit neural network
        this.fitNeuralNetwork(net, dataset);
    }

    /**
     * Process data from experience memory to a sampled input for the Neural Network
     *
     * @param boardSize Width of the board (squared board is assumed)
     * @param gameIndex Index of the current game
     * @return Dataset which can be used to update weights of the Neural Network
     */
    private List<DataSet> processData(int boardSize, int gameIndex) {
        // Experience replay
        int[] selectedIndices = experienceReplay(this.trainData.size(),
                gameIndex, maxSizeExperienceReplay, samplingRateExperienceReplay);

        // Convert hashmap to input for NN
        INDArray[] data = hashMapToINDArray(this.trainData, selectedIndices, boardSize);

        // Create dataset
        List<DataSet> dataset = new DataSet(data[0], data[1]).asList();
        Collections.shuffle(dataset, rng);

        return dataset;
    }

    /**
     * Updates the weights of the Neural Network
     *
     * @param net     Neural Network class
     * @param dataset Dataset which can be used to fit neural network
     */
    private void fitNeuralNetwork(MultiLayerNetwork net, List<DataSet> dataset) {
        // Create iterator
        DataSetIterator iterator = new ListDataSetIterator<>(dataset, this.batchSize);

        // Train network
        for (int k = 0; k < this.nEpochs; k++) {
            iterator.reset();
            net.fit(iterator);
        }
    }

    /**
     * Samples indices based on experience replay
     *
     * @param size         Size of experience memory
     * @param gameIndex    Index of current game
     * @param maxSize      Maximum number of games in experience replay
     * @param samplingRate The rate of the total pairs that are sampled from experience replay
     * @return Sample from experience replay
     */
    private int[] experienceReplay(int size, int gameIndex, int maxSize, float samplingRate) {
        // If index is too old, remove
        if (this.trainData.get(this.trainData.keySet().toArray()[0]).index < (gameIndex - maxSize + 1)) {
            keepMaxSizeGames(this.trainData, gameIndex, maxSize);
            size = this.trainData.size();
        }

        // Get the size to sample from
        int sampleSize = (int) (size * samplingRate);

        // Sample from total size (shuffle and take first maxSize * samplingRate)
        List<Integer> sampledIndices = IntStream.range(0, size).boxed().collect(toList());
        Collections.shuffle(sampledIndices);

        // Convert to array
        int[] indices = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            indices[i] = sampledIndices.get(i);
        }

        return indices;
    }

    /**
     * Remove all old games, only keep the last few games (based on given game index)
     *
     * @param linkedHashMap Experience memory
     * @param gameIndex     Index of current game
     * @param maxSize       Total number of games allowed in experience memory
     */
    private void keepMaxSizeGames(LinkedHashMap<String, TrainingSample> linkedHashMap, int gameIndex, int maxSize) {
        int removeGameIndex = gameIndex - maxSize + 1;
        int removeIndex = 0;
        Object[] keys = linkedHashMap.keySet().toArray();
        for (int i = 0; i < keys.length; i++) {
            if (linkedHashMap.get(keys[i]).index >= removeGameIndex) {
                removeIndex = i;
                break;
            }
        }

        // Remove the oldest n elements, to keep the newest w.r.t. the maximum size
        linkedHashMap.keySet().removeAll(
                Arrays.asList(linkedHashMap.keySet().toArray())
                        .subList(0, removeIndex)
        );
    }

    /**
     * Mirrors the matrix representation of the board vertically
     *
     * @param toMirror Original matrix representation of the board (NN input)
     * @return Vertical mirrored matrix representation of the board
     */
    private INDArray mirrorBoardVertically(INDArray toMirror) {
        long[] shape = toMirror.shape();
        int batches = (int) shape[0];
        int boardSize = (int) shape[2];
        INDArray verticalBoard = Nd4j.zeros(shape);
        for (int b = 0; b < batches; b++) {
            for (int p = 0; p < numPlayers; p++) {
                for (int i = 0; i < boardSize; i++) {
                    verticalBoard.slice(b).slice(p)
                            .putColumn(boardSize - i - 1, toMirror.slice(b).slice(p).getColumn(i));
                }
            }
        }

        return verticalBoard;
    }

    /**
     * Mirrors the matrix representation of the board with respect to the player perspective
     *
     * @param toMirror Original matrix representation of the board (NN input)
     * @return Mirrored matrix representation of the board with respect to the player perspective
     */
    private INDArray mirrorPlayerPerspective(INDArray toMirror) {
        long[] shape = toMirror.shape();
        int batches = (int) shape[0];
        int boardSize = (int) shape[2];
        INDArray switchedPerspectiveBoard = Nd4j.zeros(shape);
        for (int b = 0; b < batches; b++) {
            for (int p = 0; p < this.numPlayers; p++) {
                for (int i = 0; i < boardSize; i++) {
                    switchedPerspectiveBoard.slice(b).slice(this.numPlayers - p - 1)
                            .putRow(boardSize - i - 1, toMirror.slice(b).slice(p).getRow(i));
                }
            }
        }

        return switchedPerspectiveBoard;
    }

    /**
     * Saves a checkpoint of the Neural Network and experience memory
     *
     * @param net Neural Network class
     * @throws IOException
     */
    public void saveCheckPoint(MultiLayerNetwork net) throws IOException {
        // Save network
        net.save(this.checkpointFile, true);

        // Save training data
        try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
                new FileOutputStream(this.checkpointTrainData))) {
            trainExamplesOutput.writeObject(this.trainData);
        }
    }

    /**
     * Saves a checkpoint of the Neural Network with a given string and experience memory
     *
     * @param net                Neural Network class
     * @param checkpointFileName File name of the saved neural network
     * @throws IOException
     */
    public void saveCheckPoint(MultiLayerNetwork net, String checkpointFileName) throws IOException {
        net.save(new File(checkpointFileName), true);

        try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
                new FileOutputStream(this.checkpointTrainData))) {
            trainExamplesOutput.writeObject(this.trainData);
        }
    }

    /**
     * Saves final Neural Network. The checkpoint of the neural networ will be deleted.
     *
     * @param net         Neural network to be saved
     * @param FileName    File name of the saved neural network
     * @param saveUpdater Indicates if the updater needs to be saved (the updater is required to continue training)
     * @throws IOException
     */
    public void saveFinalModel(MultiLayerNetwork net, String FileName, boolean saveUpdater) throws IOException {
        this.saveFinalModel(net, FileName, saveUpdater, true);
    }

    /**
     * Saves final Neural Network, while optionally deleting the checkpoint of the neural network
     *
     * @param net              Neural network to be saved
     * @param FileName         File name of the saved neural network
     * @param saveUpdater      Indicates if the updater needs to be saved (the updater is required to continue training)
     * @param deleteCheckpoint Indicates if the checkpoint of the neural network should be deleted
     * @throws IOException
     */
    public void saveFinalModel(MultiLayerNetwork net, String FileName,
                               boolean saveUpdater, boolean deleteCheckpoint) throws IOException {
        net.save(new File(FileName), saveUpdater);

        if (deleteCheckpoint) {
            this.checkpointFile.delete();
        }
    }

    /**
     * Pretrains a neural network by generating random game positions. This is done by playing with two random bots
     * until a terminal game position has been reached. A GameStateEvaluator is used to evaluate these states, which
     * be the target values of the neural network.
     *
     * @param gameName     Name of the game (as used by Ludii)
     * @param networkType  Neural Network architecture
     * @param learningRate The learning rate during training
     * @param numSamples   Number of samples (game position / pairs) to create (size of the created dataset)
     * @param patienceData Maximum number of times the game position generator needs to create a new random
     *                     board before terminating (since the created game position already exists)
     * @param dataPath     Path to pre-generated dataset to use (optionally)
     * @param saveData     Indicates if data should be saved after being generated
     * @throws IOException
     */
    public void pretrainNetwork(String gameName, NetworkType networkType, float learningRate,
                                int numSamples, int patienceData,
                                String dataPath, boolean saveData) throws IOException {
        // Create network
        System.out.println("Create a new neural network.");
        Game game = GameLoader.loadGameFromName(gameName + ".lud");
        final int boardSize = (int) Math.sqrt(game.board().numSites());
        MultiLayerNetwork newNet = this.createNetwork(game, networkType, learningRate);
        System.out.println("The network has been created: \n" + newNet.summary());

        HashMap<String, TrainingSample> createdData = null;
        // Create dataset
        if (dataPath == null) {
            System.out.println("Starting to create a dataset of final game states.");
            createdData = this.createFinalBoardHashMap(gameName, numSamples, patienceData);
            System.out.println("All games have been played.");
        } else {
            System.out.println("Load dataset...");
            try (ObjectInputStream createdExamples = new ObjectInputStream(new FileInputStream(dataPath))) {
                Object readObject = createdExamples.readObject();

                createdData = (HashMap<String, TrainingSample>) readObject;

                System.out.printf("Restored train examples from %s with %d train examples.\n",
                        dataPath,
                        createdData.size());
            } catch (ClassNotFoundException e) {
                System.out.println("Train examples could not be restored. Continue with empty train examples history.");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        assert createdData != null;
        INDArray[] data = this.hashMapToINDArray(createdData,
                IntStream.range(0, createdData.size()).toArray(), boardSize);
        List<DataSet> dataset = new DataSet(data[0], data[1]).asList();
        Collections.shuffle(dataset, rng);
        System.out.println("The dataset has been created.");

        // Save data
        if (saveData) {
            // Save training data
            try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
                    new FileOutputStream("pretrainData.obj"))) {
                trainExamplesOutput.writeObject(createdData);
            }

            System.out.println("The data has been saved to pretrainData.obj");
        }

        // Fit neural network
        System.out.println("The network has started training.");
        this.fitNeuralNetwork(newNet, dataset);
        System.out.println("The network has finished training.");

        // Check value for root
        this.predictRoot(gameName, newNet);

        // Save neural network
        this.saveFinalModel(newNet,
                "Network_" +
                        "bSize" + batchSize + "_" +
                        "nEp" + nEpochs + "_" +
                        "nSam" + createdData.size() + "_" +
                        new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date()) +
                        ".bin",
                true);
    }

    /**
     * Converts HashMap with TrainingSamples to input for the Neural Network based on given indices
     *
     * @param hashMap         HashMap which needs to be converted to NN input
     * @param selectedIndices The indices of the selected samples
     * @param boardSize       The width of the board (squared board is assumed)
     * @return Array of inputs which can be used as input by the NN
     */
    public INDArray[] hashMapToINDArray(HashMap<String, TrainingSample> hashMap, int[] selectedIndices, int boardSize) {
        // Extract all unique boards from the collected train data
        TrainingSample[] trainingSamples = hashMap.values().toArray(new TrainingSample[0]);

        // Create training data for current epoch by selecting the indices given
        INDArray input = Nd4j.zeros(selectedIndices.length, numPlayers,
                boardSize + 2 * padding, boardSize + 2 * padding);
        INDArray target = Nd4j.zeros(selectedIndices.length, 1);
        for (int i = 0; i < selectedIndices.length; i++) {
            input.put(new INDArrayIndex[]{NDArrayIndex.interval(i, i + 1),
                            NDArrayIndex.all(),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()},
                    Nd4j.create(trainingSamples[selectedIndices[i]].board,
                            1, numPlayers, boardSize + 2 * padding, boardSize + 2 * padding));
            target.putScalar(i, trainingSamples[selectedIndices[i]].value);
        }

        return new INDArray[]{input, target};
    }

    /**
     * Generates a dataset (HashMap) by generating random game positions. This is done by playing with two random bots
     * until a terminal game position has been reached. A GameStateEvaluator is used to evaluate these states, which
     * be the target values of the neural network. This is being done for the given gameName.
     *
     * @param gameName   Name of the game (as used by Ludii)
     * @param numSamples Number of samples (game position / pairs) to create (size of the created dataset)
     * @param patience   Maximum number of times the game position generator needs to create a new random
     *                   board before terminating (since the created game position already exists)
     * @return HashMap with random game positions
     */
    public HashMap<String, TrainingSample> createFinalBoardHashMap(String gameName, int numSamples, int patience) {
        // Create game environment
        Game game = GameLoader.loadGameFromName(gameName + ".lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);

        // Create evaluator for terminal (/final) board positions
//        GameStateEvaluator terminalEvaluator = new AdditiveDepthTerminalStateEvaluator(150);
        GameStateEvaluator terminalEvaluator = new ClassicTerminalStateEvaluator();

        // Initialise everything to keep track of final game states
        NeuralNetworkLeafEvaluator NNLE = new NeuralNetworkLeafEvaluator(game, null);
        HashMap<String, TrainingSample> contexts = new HashMap<>();

        // Initialise agents
        final List<AI> agents = new ArrayList<>();
        agents.add(null);    // insert null at index 0, because player indices start at 1
        agents.add(new RandomAI());
        agents.add(new RandomAI());

        int numBoardFound = 0;
        int currentPatience = 0;
        float[] mirroredBoard;
        while (contexts.size() < numSamples) {
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

            float finalScore = terminalEvaluator.evaluate(context, 1);

            // Save a copy of the context (and the mirrored board)
            INDArray board = NNLE.boardToInput(new Context(context));
            contexts.remove(Arrays.toString(board.data().asFloat()));
            contexts.put(Arrays.toString(board.data().asFloat()),
                    new TrainingSample(board.data().asFloat(), finalScore, 0));

            // If a new board has been added
            if (contexts.size() > numBoardFound) {
                // Add the board based on symmetry as well
                mirroredBoard = this.mirrorBoardVertically(board).data().asFloat();
                contexts.put(Arrays.toString(mirroredBoard),
                        new TrainingSample(mirroredBoard, finalScore, 0));

                // Update user
                numBoardFound += 2;
                System.out.printf("Created %d/%d positions.\n", numBoardFound, numSamples);

                // Reset patience
                currentPatience = 0;
            } else {
                // Update patience, if to many times, no new update, return everything
                currentPatience++;

                if (currentPatience >= patience) {
                    System.out.printf("After %d attempts, no new boards have been found.\n" +
                            "Total amount of final boards created: %d.\n", currentPatience, contexts.size());
                    return contexts;
                }
            }
        }

        return contexts;
    }

    /**
     * Predicts the value of the starting position based on the Neural Network and game name given
     *
     * @param gameName Name of the game (as used by Ludii)
     * @param net      Neural Network class
     */
    public void predictRoot(String gameName, MultiLayerNetwork net) {
        // Create game environment
        Game game = GameLoader.loadGameFromName(gameName + ".lud");
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);
        game.start(context);

        // Predict game state
        NeuralNetworkLeafEvaluator NNLE = new NeuralNetworkLeafEvaluator(game, null);
        System.out.printf("The prediction for the root is: %.2f.\n",
                net.output(NNLE.boardToInput(context), false).getFloat(0));
    }
}

/**
 * Single sample in created dataset which stores the board and the estimated value after searching
 */
class TrainingSample implements Serializable {

    //-------------------------------------------------------------------------

    /**
     * Array representing the board of the given game position
     */
    public float[] board;

    /**
     * Estimated value of given game position
     */
    public float value;

    /**
     * Index of the game the sample is created in
     */
    public int index;

    //-------------------------------------------------------------------------

    /**
     * Constructor to create a TrainingSample with the board, value and index as input.
     *
     * @param board Array representing the board of the given game position
     * @param value Estimated value of given game position
     * @param index Index of the game the sample is created in
     */
    public TrainingSample(float[] board, float value, int index) {
        this.board = board;
        this.value = value;
        this.index = index;
    }

    /**
     * Converts class to string basec on board and value
     *
     * @return String representing the TrainingSample
     */
    @Override
    public String toString() {
        return Arrays.toString(this.board) + ", " + this.value;
    }
}
