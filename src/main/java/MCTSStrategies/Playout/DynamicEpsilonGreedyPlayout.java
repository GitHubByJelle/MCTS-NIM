package MCTSStrategies.Playout;

import Evaluator.GameStateEvaluator;
import MCTSStrategies.MoveSelector.BatchedEvaluatedMoveSelector;
import game.rules.phase.Phase;
import other.AI;
import other.context.Context;
import other.playout.PlayoutMoveSelector;
import other.trial.Trial;
import playout_move_selectors.EpsilonGreedyWrapper;
import search.mcts.MCTS;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Runs an epsilon-greedy play-out until a threshold has been reached based on a GameStateEvaluator.
 */
public class DynamicEpsilonGreedyPlayout extends EpsilonGreedyPlayout {

    //-------------------------------------------------------------------------

    /**
     * Threshold value indicating when to stop the play-out
     */
    protected float threshold;

    //-------------------------------------------------------------------------

    /**
     * Constructor to create an epsilon-greedy dynamic play-out
     *
     * @param epsilon   Probability of playing a random move (value between 0 and 1)
     * @param threshold Threshold value indicating when to stop the play-out
     */
    public DynamicEpsilonGreedyPlayout(float epsilon, float threshold) {
        super(epsilon);
        this.threshold = threshold;
    }

    /**
     * Constructor to create an epsilon-greedy dynamic play-out
     *
     * @param epsilon                Probability of playing a random move (value between 0 and 1)
     * @param threshold              Threshold value indicating when to stop the play-out
     * @param leafEvaluator          GameStateEvaluator which can be used to evaluate non-terminal game states
     *                               (when null, the same leaf evaluator as the mcts will be used)
     * @param terminalStateEvaluator GameStateEvaluator which can be used to evaluate terminal game states
     *                               (when null, the same leaf evaluator as the mcts will be used)
     */
    public DynamicEpsilonGreedyPlayout(float epsilon, float threshold,
                                       GameStateEvaluator leafEvaluator,
                                       GameStateEvaluator terminalStateEvaluator) {
        super(epsilon);
        this.threshold = threshold;

        this.leafEvaluator = leafEvaluator;
        this.terminalStateEvaluator = terminalStateEvaluator;

        this.moveSelector.setLeafEvaluator(this.leafEvaluator);
        this.moveSelector.setTerminalStateEvaluator(terminalStateEvaluator);
    }

    /**
     * Runs the epsilon-greedy dynamic play-out on the current context.
     *
     * @param mcts    Ludii's MCTS class
     * @param context Ludii's context class representing the game position
     * @return Ludii's trial class with after performing the play-out on the given game position
     */
    @Override
    public Trial runPlayout(MCTS mcts, Context context) {
        return dynamicPlayout(context, (List) null, -1,
                new EpsilonGreedyWrapper(this.moveSelector, this.epsilon),
                -1, this.playoutTurnLimit, ThreadLocalRandom.current());
    }

    /**
     * Performs a dynamic epsilon-greedy play-out given a Move Selector (using the GameStateEvaluator) and a
     * game position
     *
     * @param context              Ludii's context class representing the game position
     * @param ais                  The ais used to perform the play-out
     * @param thinkingTime         Thinking time in seconds per move (can be set to -1 to prevent limits)
     * @param playoutMoveSelector  Ludii's move selector which selects a move for every context.
     * @param maxNumBiasedActions  Maximum number of actions during the play-out which are determined by the move selector
     * @param maxNumPlayoutActions Maximum number of actions during the play-out
     * @param random               Random generator to generate pseudorandom numbers
     * @return Ludii's trial class with after performing the play-out on the given game position
     */
    private Trial dynamicPlayout(final Context context, final List<AI> ais, final double thinkingTime,
                                 final PlayoutMoveSelector playoutMoveSelector, final int maxNumBiasedActions,
                                 final int maxNumPlayoutActions, final Random random) {
        // Initialise needed variables
        Random rng = random != null ? random : ThreadLocalRandom.current();
        Trial trial = context.trial();
        int numStartMoves = trial.numMoves();

        // Determine value of current value
        double estimatedValue = this.leafEvaluator.evaluate(context,
                context.state().playerToAgent(context.state().mover()));

        // As long as the game position is not terminal, there are moves left, and the estimated values is between
        // the bounds, continue the play-out.
        while (!trial.over() && (maxNumPlayoutActions < 0 || trial.numMoves() - numStartMoves < maxNumPlayoutActions) &&
                (Math.abs(estimatedValue) < this.threshold)) {
            // Determine the maximum number of allowed biased actions
            int numAllowedBiasedActions;
            if (maxNumBiasedActions >= 0) {
                numAllowedBiasedActions = Math.max(0, maxNumBiasedActions - (trial.numMoves() - numStartMoves));
            } else {
                numAllowedBiasedActions = maxNumBiasedActions;
            }

            // Perform a single ply
            Phase phase = context.game().rules().phases()[context.state().currentPhase(context.state().mover())];
            if (phase.playout() != null) {
                trial = phase.playout().playout(context, ais, thinkingTime, playoutMoveSelector, numAllowedBiasedActions, 1, (Random) rng);
            } else if (context.game().mode().playout() != null) {
                trial = context.game().mode().playout().playout(context, ais, thinkingTime, playoutMoveSelector, numAllowedBiasedActions, 1, (Random) rng);
            } else {
                trial = context.model().playout(context, ais, thinkingTime, playoutMoveSelector, numAllowedBiasedActions, 1, (Random) rng);
            }

            // Determine the estimated value for the new game positions
            if (!trial.over()) {
                estimatedValue = this.leafEvaluator.evaluate(context,
                        context.state().playerToAgent(context.state().mover()));
            }
        }

        return trial;
    }

    /**
     * Sets the leaf evaluator to the move selector used when not already set
     *
     * @param leafEvaluator GameStateEvaluator which can be used to evaluate non-terminal game states
     */
    @Override
    public void setLeafEvaluator(GameStateEvaluator leafEvaluator) {
        if (this.leafEvaluator == null) {
            this.leafEvaluator = leafEvaluator;
            this.moveSelector.setLeafEvaluator(leafEvaluator);
        }
    }

    /**
     * Sets the terminal state evaluator to the move selector used when not already set
     *
     * @param terminalStateEvaluator GameStateEvaluator which can be used to evaluate terminal game states
     */
    @Override
    public void setTerminalStateEvaluator(GameStateEvaluator terminalStateEvaluator) {
        if (this.terminalStateEvaluator == null) {
            this.terminalStateEvaluator = terminalStateEvaluator;
            this.moveSelector.setTerminalStateEvaluator(terminalStateEvaluator);
        }
    }

    /**
     * Creates a batched move selector when the game state evaluators are not already set. Please note, this need to be
     * performed before setting the GameStateEvaluator to the move selector.
     */
    @Override
    public void createBatchedMoveSelector() {
        if (this.leafEvaluator == null && this.terminalStateEvaluator == null) {
            this.moveSelector = new BatchedEvaluatedMoveSelector();
        }
    }
}
