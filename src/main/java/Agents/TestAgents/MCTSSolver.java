package Agents;

import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Wrapper.debugFinalSelectionWrapper;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.playout.RandomPlayout;
import search.mcts.selection.UCB1;

/**
 * MCTS search algorithm with UCB1, random play-outs, monte carlo backpropagation, and robust child with
 * the solver enabled.
  */
public class MCTSSolver extends MCTS {

    /**
     * Constructor requiring no inputs (exploration=sqrt(2), and 6 threads)
     */
    public MCTSSolver() {
        // Original
        super(new UCB1(), new RandomPlayout(-1),
                new MonteCarloBackprop(),
                new RobustChild());
//                new debugFinalSelectionWrapper(new RobustChild()));
//                new IterationWrapper(new RobustChild()));
        this.friendlyName = "MCTS-Solver";
        this.setNumThreads(6);
        this.setUseSolver(true);
    }
}