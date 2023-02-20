package Training;

import org.nd4j.jita.conf.CudaEnvironment;
import utils.Enums.NetworkType;
import utils.propertyLoader;

import java.io.IOException;
import java.util.Properties;

/**
 * Main class to pretrain a Neural Network
 */
public class Pretraining {
    /**
     * Main class to pretrain a Neural network based on the configuration in the .properties file
     * @param args All inputs for pretraining. A .proporties file with the following inputs is expected
     *             (see configurations/templatePretraining.proporties):
     *             String gameName: Name of the game (as used by Ludii)
     *             int batchSize: The number of pairs in a single batch during training
     *             int nEpochs: The number of epochs on the created data
     *             float learningRate: The learning rate during training
     *             int numSamples: Number of samples (game position / pairs) to create (size of the created dataset)
     *             int patienceData: Maximum number of times the game position generator needs to create a new random
     *                               board before terminating (since the created game position already exists)
     *             String dataPath: Path to pre-generated dataset to use (optionally)
     *             boolean saveData: Indicates if data should be saved after being generated
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        // Check if os is Linux (server), if so use both GPU
        if (System.getProperty("os.name") == "Linux"){
            CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
        }

        // Load properties from file (or use template)
        Properties props;
        if (args.length == 0) {
            System.out.println("The basic configuration file has been loaded. No input detected.");
            props = propertyLoader.getPropertyValues("configurations\\templatePretraining.properties");
        } else {
            props = propertyLoader.getPropertyValues(args[0]);
        }

        // Extract all information
        final String gameName = props.getProperty("gameName");
        final int batchSize = Integer.parseInt(props.getProperty("batchSize"));
        final int nEpochs = Integer.parseInt(props.getProperty("nEpochs"));
        final float learningRate = Float.parseFloat(props.getProperty("learningRate"));
        final int numSamples = Integer.parseInt(props.getProperty("numSamples"));
        final int patienceData = Integer.parseInt(props.getProperty("patienceData"));
        final String dataPath = props.getProperty("dataPath");
        final boolean saveData = Boolean.parseBoolean(props.getProperty("saveData"));

        // Create a Learning Manager (no dataselection, maxSizeExperienceReplay or samplingRateExperienceReplay is needed)
        LearningManager learningManager = new LearningManager(null, nEpochs, batchSize,
                0, 0);

        // Pretrains neural network
        learningManager.pretrainNetwork(gameName, NetworkType.Cohen, learningRate,
                numSamples, patienceData, dataPath, saveData);
    }
}
