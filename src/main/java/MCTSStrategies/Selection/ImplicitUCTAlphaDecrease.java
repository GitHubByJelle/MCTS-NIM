//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values, with an
 * decreasing alpha
 */
public class ImplicitUCTAlphaDecrease extends ImplicitUCTAlphaIncrease {

    /**
     * Constructor with influence of the implicit minimax value, exploration constant and slope as input
     *
     * @param initialInfluenceEstimatedMinimax Initial influence of the implicit minimax value
     * @param explorationConstant Exploration constant
     * @param slope Slope of decrease of alpha
     */
    public ImplicitUCTAlphaDecrease(double initialInfluenceEstimatedMinimax, double explorationConstant, double slope) {
        super(initialInfluenceEstimatedMinimax, explorationConstant, slope);
    }

    /**
     * Adjust alpha to decrease over-time
     *
     * @param initialAlpha Initial influence of the estimated values
     * @param numVisits Number of visits to current node
     * @return Adjusted alpha
     */
    protected double adjustAlpha(double initialAlpha, int numVisits){
        return Math.max(0, initialAlpha - this.slope * numVisits * initialAlpha);
    }
}
