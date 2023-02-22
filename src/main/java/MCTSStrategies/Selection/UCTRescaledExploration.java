//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import MCTSStrategies.Rescaler.Softmax;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on UCT. However, instead of multiplying all children with the
 * same exploration constant, a softmax with temperature C/n_c is used to transform the estimated values to
 * probabilities which are multiplied with the exploration.
 */
public class UCTRescaledExploration extends ImplicitUCT {

    //-------------------------------------------------------------------------

    /**
     * Softmax rescaler
     */
    Softmax rescaler;

    //-------------------------------------------------------------------------

    /**
     * Constructor with exploration constant and softmax rescaler as input
     *
     * @param explorationConstant Exploration constant
     * @param rescaler            Rescaler used to rescale the estimated values
     */
    public UCTRescaledExploration(double explorationConstant, Softmax rescaler) {
        this.explorationConstant = explorationConstant;
        this.rescaler = rescaler;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on UCT with
     * modified exploration. The exploration constant is multiplied with probabilities determined by a
     * softmax with temperature C/n_c on the estimated values.
     *
     * @param mcts    Ludii's MCTS class
     * @param current node representing the current game state
     * @return The index of next "best" move
     */
    public int select(MCTS mcts, BaseNode current) {
        // Initialise needed variables
        int bestIdx = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        int numBestFound = 0;
        double parentVisits = (double) Math.max(1, current.sumLegalChildVisits());
        double parentLog = Math.log(parentVisits);
        int numChildren = current.numLegalMoves();
        State state = current.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = current.valueEstimateUnvisitedChildren(moverAgent);

        // Determine all estimated values
        double[] estimatedValues = new double[numChildren];
        for (int i = 0; i < numChildren; i++) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                estimatedValues[i] = ((implicitNode) current).getInitialEstimatedValue(i); // Own perspective
            } else {
                estimatedValues[i] = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }
        }

        // Determine exploration probabilities based on the softmax
        double[] explorationProbs = this.rescaler.rescale(estimatedValues, explorationConstant / parentVisits);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        for (int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = explorationProbs[i] * Math.sqrt(parentLog);
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = explorationProbs[i] * Math.sqrt(parentLog / (double) numVisits);
            }

            double ucb1Value = exploit + explore;

            if (ucb1Value > bestValue) {
                bestValue = ucb1Value;
                bestIdx = i;
                numBestFound = 1;
            } else if (ucb1Value == bestValue) {
                int randomInt = ThreadLocalRandom.current().nextInt();
                ++numBestFound;
                if (randomInt % numBestFound == 0) {
                    bestIdx = i;
                }
            }
        }

        return bestIdx;
    }
}
