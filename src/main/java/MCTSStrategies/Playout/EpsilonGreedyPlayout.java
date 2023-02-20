package MCTSStrategies.Playout;

import Evaluator.GameStateEvaluator;
import MCTSStrategies.MoveSelector.EvaluatedMoveSelector;
import MCTSStrategies.MoveSelector.BatchedEvaluatedMoveSelector;
import game.Game;
import other.context.Context;
import other.trial.Trial;
import playout_move_selectors.EpsilonGreedyWrapper;
import search.mcts.MCTS;
import search.mcts.playout.HeuristicPlayout;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class EpsilonGreedyPlayout extends HeuristicPlayout {

    //-------------------------------------------------------------------------

    /** Probability of playing a random move (value between 0 and 1) */
    protected float epsilon;

    /** GameStateEvaluator which can be used to evaluate non-terminal game states */
    protected GameStateEvaluator leafEvaluator;

    /** GameStateEvaluator which can be used to evaluate terminal game states */
    protected GameStateEvaluator terminalStateEvaluator;

    /** Ludii's move selector class which selects a move for every context based on the GameStateEvaluator */
    public EvaluatedMoveSelector moveSelector = new EvaluatedMoveSelector();

    //-------------------------------------------------------------------------

    /**
     * Constructor for the Epsilon Greedy playout
     *
     * @param epsilon Probability of playing a random move (value between 0 and 1)
     */
    public EpsilonGreedyPlayout(float epsilon){
        super();
        this.epsilon = epsilon;
    }

    /**
     * Constructor for the Epsilon Greedy playout
     *
     * @param epsilon Probability of playing a random move (value between 0 and 1)
     * @param playoutTurnLimit Maximum number of plies in play-out (-1 indicates no limit)
     */
    public EpsilonGreedyPlayout(float epsilon, int playoutTurnLimit){
        super();
        this.epsilon = epsilon;
        this.playoutTurnLimit = playoutTurnLimit;
    }

    /**
     * Runs the epsilon-greedy play-out on the current context.
     *
     * @param mcts Ludii's MCTS class
     * @param context Ludii's context class representing the game position
     * @return Ludii's trial class with after performing the play-out on the given game position
     */
    @Override
    public Trial runPlayout(MCTS mcts, Context context) {
        return context.game().playout(context, (List)null, 1.0,
                new EpsilonGreedyWrapper(this.moveSelector, this.epsilon),
                -1, this.playoutTurnLimit, ThreadLocalRandom.current());
    }

    /**
     * Initialiser for the AI (doesn't do anything, but is required)
     *
     * @param game Ludii's game class
     * @param playerID ID of current player
     */
    public void initAI(Game game, int playerID) {
    }

    /**
     * Sets the leaf evaluator to the move selector used
     *
     * @param leafEvaluator GameStateEvaluator which can be used to evaluate non-terminal game states
     */
    public void setLeafEvaluator(GameStateEvaluator leafEvaluator) {
        this.leafEvaluator = leafEvaluator;
        this.moveSelector.setLeafEvaluator(leafEvaluator);
    }

    /**
     * Sets the terminal state evaluator to the move selector used
     *
     * @param terminalStateEvaluator GameStateEvaluator which can be used to evaluate terminal game states
     */
    public void setTerminalStateEvaluator(GameStateEvaluator terminalStateEvaluator){
        this.terminalStateEvaluator = terminalStateEvaluator;
        this.moveSelector.setTerminalStateEvaluator(terminalStateEvaluator);
    }

    /**
     * Creates a batched move selector. Please note, this need to be performed before setting the GameStateEvaluator
     * to the move selector.
     */
    public void createBatchedMoveSelector(){
        this.moveSelector = new BatchedEvaluatedMoveSelector();
    }
}
