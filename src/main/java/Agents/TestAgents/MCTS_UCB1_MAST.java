package Agents.MCTSTest;

import Agents.MCTS;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.MAST;
import search.mcts.selection.UCB1;

/**
 * MCTS implementation with UCB1 and MAST
 */
public class MCTS_UCB1_MAST extends MCTS {
    /**
     * Constructor with no inputs
     */
    public MCTS_UCB1_MAST() {
        // Original
        super(new UCB1(),
                new MAST(-1, .05f),
                new MonteCarloBackprop(),
                new RobustChild());

        this.friendlyName = "MCTS-Test";
        this.setNumThreads(16);
    }
}