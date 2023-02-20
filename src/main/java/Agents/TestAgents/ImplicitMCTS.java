package Agents;

import Evaluator.*;
import MCTSStrategies.Backpropagation.DynamicEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.FinalMoveSelection.SecureChild;
import MCTSStrategies.Playout.DynamicEpsilonGreedyPlayout;
import MCTSStrategies.Selection.ImplicitUCT;
import MCTSStrategies.Wrapper.IterationWrapper;
import MCTSStrategies.Wrapper.debugFinalSelectionWrapper;
import game.Game;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.RandomPlayout;

/**
 * MCTS search algorithm with Implicit UCT, epsilon-greedy play-out with dynamic early termination (when |x| > bound,
 * backpropagate win or loss), and robust child. It uses the Maarten Schadds Evaluation function, as
 * proposed in Lanctot, M., Winands, M. H., Pepels, T., & Sturtevant, N. R. (2014, August). Monte Carlo
 * tree search with heuristic evaluations using implicit minimax backups. In 2014 IEEE Conference on
 * Computational Intelligence and Games (pp. 1-8). IEEE.
 */
public class ImplicitMCTS extends MCTS {

    /**
     * Constructor requiring no inputs (exploration=sqrt(2), influence estimated value = 0.4, bound = 0.4,
     * epsilon = 0.1, and 12 threads)
     */
    public ImplicitMCTS() {
        super(new ImplicitUCT(.4, Math.sqrt(2)), new DynamicEpsilonGreedyPlayout(.1f, .4f),
                new DynamicEarlyTerminationBackprop(.4f), //new RobustChild());
                new RobustChild());

        this.setNumThreads(12);
    }

    /**
     * Perform desired initialisation before starting to play a game
     * Initialise the parent and both GameStateEvaluators
     *
     * @param game The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    public void initAI(Game game, int playerID){
        super.initParent(game, playerID);

        this.setLeafEvaluator(new TanhEvaluatorWrapper(new MSLeafEvaluator(game),
                60, 100, -100), game);
        this.setTerminalStateEvaluator(new ClassicTerminalStateEvaluator());
    }
}