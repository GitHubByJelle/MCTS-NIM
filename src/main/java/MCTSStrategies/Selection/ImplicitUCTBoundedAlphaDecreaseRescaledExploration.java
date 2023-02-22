//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Rescaler.Softmax;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values, with a
 * decreasing alpha. However, instead of multiplying all children with the same exploration constant, a softmax
 * with temperature C/n_c is used to transform the estimated values to probabilities which are multiplied with the
 * exploration.
 */
public class ImplicitUCTBoundedAlphaDecreaseRescaledExploration extends ImplicitUCTAlphaDecreaseRescaledExploration {

    //-------------------------------------------------------------------------

    /**
     * Minimum bound for influence of the estimates values
     */
    protected double minimumInfluenceEstimatedMinimax;

    //-------------------------------------------------------------------------

    /**
     * Constructor with influence of the implicit minimax value, exploration constant, rescaler and slope as input
     *
     * @param initialInfluenceEstimatedMinimax Initial influence of the implicit minimax value
     * @param explorationConstant              Exploration constant
     * @param rescaler                         Softmax rescaler
     * @param slope                            Slope of increase of alpha
     * @param minimumInfluenceEstimatedMinimax Minimum bound for influence of the estimates values
     */
    public ImplicitUCTBoundedAlphaDecreaseRescaledExploration(double initialInfluenceEstimatedMinimax, double explorationConstant,
                                                              Softmax rescaler, double slope, double minimumInfluenceEstimatedMinimax) {
        super(initialInfluenceEstimatedMinimax, explorationConstant, rescaler, slope);

        this.minimumInfluenceEstimatedMinimax = minimumInfluenceEstimatedMinimax;
    }

    /**
     * Adjust alpha to decrease over-time until a minimum value is reached
     *
     * @param initialAlpha Initial influence of the estimated values
     * @param numVisits    Number of visits to current node
     * @return Adjusted alpha
     */
    protected double adjustAlpha(double initialAlpha, int numVisits) {
        return Math.max(this.minimumInfluenceEstimatedMinimax,
                initialAlpha - this.slope * numVisits * initialAlpha);
    }
}
