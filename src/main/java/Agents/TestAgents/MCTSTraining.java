package Agents;

import Evaluator.ClassicTerminalStateEvaluator;
import Evaluator.GameStateEvaluator;
import Evaluator.MultiNeuralNetworkLeafEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import MCTSStrategies.Wrapper.TrainingPlayoutWrapper;
import MCTSStrategies.Wrapper.TrainingSelectionWrapper;
import game.Game;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import other.context.Context;
import search.mcts.backpropagation.BackpropagationStrategy;
import search.mcts.finalmoveselection.FinalMoveSelectionStrategy;
import search.mcts.nodes.BaseNode;
import search.mcts.playout.PlayoutStrategy;
import search.mcts.selection.SelectionStrategy;
import utils.Enums;
import utils.TranspositionTableLearning;

/**
 * Public abstract class for MCTS which can be used for implementing search algorithms that need to be used
 * in the descent framework. It is implemented to additionally store
 * set the required objects to all strategies which are needed for training. This allows the search algorithm to be
 * used in the descent framework
 */
public class MCTSTraining extends MCTS {
    protected Enums.DataSelection dataSelection;
    protected TranspositionTableLearning TTTraining = null;
    protected MultiLayerNetwork net;
    protected NeuralNetworkLeafEvaluator leafEvaluator;
    protected GameStateEvaluator terminalEvaluator;

    /**
     * Constructor requiring the architecture as input
     *
     * @param selectionStrategy          The used selection strategy
     * @param playoutStrategy            The used play-out strategy
     * @param backpropagationStrategy    The used backpropagation strategy
     * @param finalMoveSelectionStrategy The used final move selection strategy
     * @param net                        MultiLayerNetwork (DeepLearning4J) used for training
     */
    public MCTSTraining(SelectionStrategy selectionStrategy,
                        PlayoutStrategy playoutStrategy,
                        BackpropagationStrategy backpropagationStrategy,
                        FinalMoveSelectionStrategy finalMoveSelectionStrategy,
                        MultiLayerNetwork net) {
        super(selectionStrategy, playoutStrategy, backpropagationStrategy, finalMoveSelectionStrategy);
        this.net = net;
    }

    /**
     * Setter for the Transposition Table used for storing the trainings data to all required objects
     *
     * @param tt Transposition Table used for storing the trainings data
     */
    public void setTTTraining(TranspositionTableLearning tt) {
        // Saves to mcts
        this.TTTraining = tt;

        // Saves to final move selection (to save all nodes before playing the actual move)
        if (this.finalMoveSelectionStrategy instanceof TrainingSelectionWrapper) {
            ((TrainingSelectionWrapper) this.finalMoveSelectionStrategy).setTTTraining(tt);
        } else
            throw new RuntimeException("Final move selection doesn't contain " + TrainingSelectionWrapper.class.getName());

        // Saves to playout (when performing a pay-out)
        if (this.playoutStrategy instanceof TrainingPlayoutWrapper) {
            ((TrainingPlayoutWrapper) this.playoutStrategy).setTTTraining(tt);
        }
    }

    /**
     * Getter for the Transposition Table used for storing the trainings data
     *
     * @return Transposition Table used for storing the trainings data
     */
    public TranspositionTableLearning getTTTraining() {
        return TTTraining;
    }

    /**
     * Setter for the enum for the data selection strategy used from the descent framework
     *
     * @param dataSelection Enum for the data selection strategy used from the descent framework
     */
    public void setDataSelection(Enums.DataSelection dataSelection) {
        this.dataSelection = dataSelection;
        if (this.finalMoveSelectionStrategy instanceof TrainingSelectionWrapper) {
            ((TrainingSelectionWrapper) this.finalMoveSelectionStrategy).setDataSelection(dataSelection);
        } else
            throw new RuntimeException("Final move selection doesn't contain " + TrainingSelectionWrapper.class.getName());
    }

    /**
     * Adds the final (terminal) state of an actual game to the Transposition Table with trainings data
     *
     * @param terminalContext Ludii's context class representing a final terminal game position of the actual game
     */
    public void addTerminalStateToTT(Context terminalContext) {
        INDArray inputNN = this.leafEvaluator.boardToInput(terminalContext);
        float outputScore = this.terminalEvaluator.evaluate(terminalContext, 1);
        long zobrist = terminalContext.state().fullHash(terminalContext);
        this.TTTraining.store(zobrist, outputScore, 999, inputNN.data().asFloat());
    }

    /**
     * Perform desired initialisation before starting to play a game
     *
     * @param game     The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    public void initAI(Game game, int playerID) {
        super.initParent(game, playerID);

        this.leafEvaluator = new MultiNeuralNetworkLeafEvaluator(game, this.net, this.numThreads);
        this.setLeafEvaluator(this.leafEvaluator, game);

        this.terminalEvaluator = new ClassicTerminalStateEvaluator();
        this.setTerminalStateEvaluator(this.terminalEvaluator);
    }

    /**
     * Never stop searching when learning, since all data can be used for learning
     *
     * @param rootThisCall Root node of current call
     * @param mover        ID of the player to move
     * @return True if the search should terminate the search early
     */
    @Override
    protected boolean earlyStop(BaseNode rootThisCall, int mover) {
        return this.stop;
    }
}