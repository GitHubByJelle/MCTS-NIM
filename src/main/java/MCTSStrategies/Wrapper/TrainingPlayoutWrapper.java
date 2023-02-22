package MCTSStrategies.Wrapper;

import Evaluator.GameStateEvaluator;
import MCTSStrategies.MoveSelector.TrainerEvaluatedMoveSelector;
import MCTSStrategies.Playout.EpsilonGreedyPlayout;
import game.Game;
import other.context.Context;
import other.trial.Trial;
import search.mcts.MCTS;
import search.mcts.playout.PlayoutStrategy;
import utils.TranspositionTableLearning;

/**
 * Wrapper which manages all needed variables used for storing trainings data to the EpsilonGreedyPlayout strategy used
 */
public class TrainingPlayoutWrapper implements PlayoutStrategy {

    //-------------------------------------------------------------------------

    /**
     * EpsilonGreedyPlayout strategy
     */
    public EpsilonGreedyPlayout playoutStrategy;

    //-------------------------------------------------------------------------

    /**
     * Constructor
     *
     * @param playoutStrategy EpsilonGreedyPlayout strategy
     */
    public TrainingPlayoutWrapper(EpsilonGreedyPlayout playoutStrategy) {
        this.playoutStrategy = playoutStrategy;
    }

    /**
     * Runs the play-out based on the EpsilonGreedyPlayout strategy
     *
     * @param mcts    Ludii's MCTS class
     * @param context Ludii's context class representing the game position
     * @return Ludii's trial class with after performing the play-out on the given game position
     */
    @Override
    public Trial runPlayout(MCTS mcts, Context context) {
        return this.playoutStrategy.runPlayout(mcts, context);
    }

    /**
     * Checks if the given play-out supports the game
     *
     * @param game Ludii's game class
     * @return Boolean if play-out strategy supports game
     */
    @Override
    public boolean playoutSupportsGame(Game game) {
        return this.playoutStrategy.playoutSupportsGame(game);
    }

    /**
     * Flags for data this Backpropagation wants to track.
     *
     * @return Additional flags for data this Backpropagation wants to track.
     */
    @Override
    public int backpropFlags() {
        return this.playoutStrategy.backpropFlags();
    }

    /**
     * Copies the customise method of the initialised play-out strategy
     *
     * @param strings inputs to customise
     */
    @Override
    public void customise(String[] strings) {
        this.playoutStrategy.customise(strings);
    }

    /**
     * Initialises the play-out based on the given play-out strategy
     *
     * @param game     Ludii's game class
     * @param playerID ID of current player
     */
    public void initAI(Game game, int playerID) {
        this.playoutStrategy.initAI(game, playerID);
    }

    /**
     * Sets the leaf evaluator to the given play-out strategy
     *
     * @param leafEvaluator GameStateEvaluator which can be used to evaluate non-terminal game states
     */
    public void setLeafEvaluator(GameStateEvaluator leafEvaluator) {
        this.playoutStrategy.setLeafEvaluator(leafEvaluator);
    }

    /**
     * Sets the terminal state evaluator to the given play-out strategy
     *
     * @param terminalStateEvaluator GameStateEvaluator which can be used to evaluate terminal game states
     */
    public void setTerminalStateEvaluator(GameStateEvaluator terminalStateEvaluator) {
        this.playoutStrategy.setTerminalStateEvaluator(terminalStateEvaluator);
    }

    /**
     * Changes the move selector of the given play-out strategy to a batched move selector
     */
    public void createBatchedMoveSelector() {
        this.playoutStrategy.moveSelector = new TrainerEvaluatedMoveSelector();
    }

    /**
     * Sets the transposition table used for learning to the given move selector of the given play-out strategy. Please
     * note, the move selector should be of class "TrainerEvaluatedMoveSelector", which can be achieved by performing
     * ".createBatchedMoveSelector()"
     *
     * @param TT Transposition Table used to store the learning data
     */
    public void setTTTraining(TranspositionTableLearning TT) {
        ((TrainerEvaluatedMoveSelector) this.playoutStrategy.moveSelector).setTTTraining(TT);
    }
}
