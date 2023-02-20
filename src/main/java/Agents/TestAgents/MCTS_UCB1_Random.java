package Agents.MCTSTest;

import Agents.MCTS;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.RandomPlayout;
import search.mcts.selection.UCB1;

/**
 * MCTS implementation with UCB1 and random play-outs
 */
public class MCTS_UCB1_Random extends MCTS {
    public MCTS_UCB1_Random() {
        super(new UCB1(),
                new RandomPlayout(-1),
                new MonteCarloBackprop(),
                new RobustChild());

        this.friendlyName = "MCTS-Test";
        this.setNumThreads(16);
    }
}