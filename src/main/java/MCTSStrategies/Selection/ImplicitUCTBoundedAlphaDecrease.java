//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values, with an
 * decreasing alpha
 */
public class ImplicitUCTBoundedAlphaDecrease extends ImplicitUCTAlphaDecrease {

    //-------------------------------------------------------------------------

    /** Minimum bound for influence of the estimates values */
    protected double minimumInfluenceEstimatedMinimax;

    //-------------------------------------------------------------------------

    /**
     * Constructor with influence of the implicit minimax value, exploration constant and slope as input
     *
     * @param initialInfluenceEstimatedMinimax Initial influence of the implicit minimax value
     * @param explorationConstant Exploration constant
     * @param slope Slope of decrease of alpha
     * @param minimumInfluenceEstimatedMinimax Minimum bound for influence of the estimates values
     */
    public ImplicitUCTBoundedAlphaDecrease(double initialInfluenceEstimatedMinimax, double explorationConstant,
                                           double slope, double minimumInfluenceEstimatedMinimax) {
        super(initialInfluenceEstimatedMinimax, explorationConstant, slope);

        this.minimumInfluenceEstimatedMinimax = minimumInfluenceEstimatedMinimax;
    }

    /**
     * Adjust alpha to decrease over-time until a minimum value is reached
     *
     * @param initialAlpha Initial influence of the estimated values
     * @param numVisits Number of visits to current node
     * @return Adjusted alpha
     */
    protected double adjustAlpha(double initialAlpha, int numVisits){
        return Math.max(this.minimumInfluenceEstimatedMinimax,
                initialAlpha - this.slope * numVisits * initialAlpha);
    }
}
