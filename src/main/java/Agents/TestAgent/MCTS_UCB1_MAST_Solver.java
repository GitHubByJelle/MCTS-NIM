package Agents.TestAgent;

import Agents.MCTS;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.MAST;
import search.mcts.selection.UCB1;

/**
 * MCTS implementation with UCB1 and MAST with solver
 */
public class MCTS_UCB1_MAST_Solver extends MCTS {
    public MCTS_UCB1_MAST_Solver() {
        // Original
        super(new UCB1(),
                new MAST(-1, .05f),
                new MonteCarloBackprop(),
                new RobustChild());

        this.friendlyName = "MCTS-Test";
        this.setNumThreads(6);
        this.setUseSolver(true);
    }
}