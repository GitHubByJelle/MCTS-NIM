package Agents.TestAgent;

import Agents.MCTS;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.RandomPlayout;
import search.mcts.selection.UCB1GRAVE;

/**
 * MCTS implementation with a combination of UCB1 and GRAVE and random play-outs
 */
public class MCTS_UCB1GRAVE_Random extends MCTS {
    /**
     * Constructor with no inputs
     */
    public MCTS_UCB1GRAVE_Random() {
        // Original
        super(new UCB1GRAVE(),
                new RandomPlayout(-1),
                new MonteCarloBackprop(),
                new RobustChild());

        this.friendlyName = "MCTS-Test";
        this.setNumThreads(6);
    }
}