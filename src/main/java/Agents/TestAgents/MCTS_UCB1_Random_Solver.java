package Agents.MCTSTest;

import Agents.MCTS;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Wrapper.debugFinalSelectionWrapper;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.RandomPlayout;
import search.mcts.selection.UCB1;

/**
 * MCTS implementation with UCB1 and random play-outs with solver
 */
public class MCTS_UCB1_Random_Solver extends MCTS {
    public MCTS_UCB1_Random_Solver() {
        super(new UCB1(),
                new RandomPlayout(-1),
                new MonteCarloBackprop(),
                new RobustChild());

        this.friendlyName = "MCTS-Test";
        this.setNumThreads(16);
        this.setUseSolver(true);
    }
}