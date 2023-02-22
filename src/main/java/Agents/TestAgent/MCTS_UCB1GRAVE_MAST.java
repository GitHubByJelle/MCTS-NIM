package Agents.TestAgent;

import Agents.MCTS;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.MAST;
import search.mcts.selection.UCB1GRAVE;

/**
 * MCTS implementation with a combination of UCB1 and GRAVE and MAST
 */
public class MCTS_UCB1GRAVE_MAST extends MCTS {
    /**
     * Constructor with no inputs
     */
    public MCTS_UCB1GRAVE_MAST() {
        // Original
        super(new UCB1GRAVE(),
                new MAST(-1, .05f),
                new MonteCarloBackprop(),
                new RobustChild());

        this.friendlyName = "MCTS-Test";
        this.setNumThreads(6);
    }
}