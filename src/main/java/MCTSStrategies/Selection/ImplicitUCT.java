//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package MCTSStrategies.Selection;

import MCTSStrategies.Node.implicitNode;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;
import search.mcts.selection.SelectionStrategy;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Selection strategy which selects the child based on a combination of UCT and minimax backpropagated values.
 * Implemented as proposed in Lanctot, M., Winands, M. H., Pepels, T., & Sturtevant, N. R. (2014, August). Monte Carlo
 * tree search with heuristic evaluations using implicit minimax backups. In 2014 IEEE Conference on
 * Computational Intelligence and Games (pp. 1-8). IEEE.
 */
public class ImplicitUCT implements SelectionStrategy {

    //-------------------------------------------------------------------------

    /**
     * Exploration constant
     */
    protected double explorationConstant;

    /**
     * Influence of the implicit minimax value
     */
    protected double influenceEstimatedMinimax;

    //-------------------------------------------------------------------------

    /**
     * Constructor with no inputs (influence=0.4 and exploration=sqrt(2))
     */
    public ImplicitUCT() {
        this(0.4, Math.sqrt(2.0));
    }

    /**
     * Constructor with influence of the implicit minimax value and exploration constant as input
     *
     * @param influenceEstimatedMinimax Influence of the implicit minimax value
     * @param explorationConstant       Exploration constant
     */
    public ImplicitUCT(double influenceEstimatedMinimax, double explorationConstant) {
        this.explorationConstant = explorationConstant;
        this.influenceEstimatedMinimax = influenceEstimatedMinimax;
    }

    /**
     * Selects the index of a child of the current node to traverse to based on implicit UCT
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
        double parentLog = Math.log((double) Math.max(1, current.sumLegalChildVisits()));
        int numChildren = current.numLegalMoves();
        State state = current.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = current.valueEstimateUnvisitedChildren(moverAgent);

        // For all children, determine child with highest uct value
        // Ties are broken at random
        double exploit;
        double explore;
        double estimatedValue;
        for (int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) current.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                estimatedValue = ((implicitNode) current).getInitialEstimatedValue(i); // Own perspective
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double) numVisits);
                estimatedValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }

            double uctValue = (1 - this.influenceEstimatedMinimax) * exploit +
                    this.influenceEstimatedMinimax * estimatedValue +
                    this.explorationConstant * explore;

//            if (current.parent() == null){
//                System.out.printf("%d) win rate: %.4f, heuristic: %.4f, explore: %.4f, ucb: %.4f\n",
//                        i+1, exploit, estimatedValue, explore, uctValue);
//            }

            if (uctValue > bestValue) {
                bestValue = uctValue;
                bestIdx = i;
                numBestFound = 1;
            } else if (uctValue == bestValue) {
                int randomInt = ThreadLocalRandom.current().nextInt();
                ++numBestFound;
                if (randomInt % numBestFound == 0) {
                    bestIdx = i;
                }
            }
        }

//        if (current.parent() == null){
//            System.out.printf("Selected: %d\n\n", bestIdx+1);
//        }

        return bestIdx;
    }

    /**
     * @return Flags indicating stats that should be backpropagated
     */
    public int backpropFlags() {
        return 0;
    }

    /**
     * @return Flags indicating special things we want to do when expanding nodes
     */
    public int expansionFlags() {
        return 0;
    }

    /**
     * Customize the selection strategy based on a list of given string inputs
     *
     * @param inputs indicating what to customise
     */
    public void customise(String[] inputs) {
        if (inputs.length > 1) {
            for (int i = 1; i < inputs.length; ++i) {
                String input = inputs[i];
                if (input.startsWith("explorationconstant=")) {
                    this.explorationConstant = Double.parseDouble(input.substring("explorationconstant=".length()));
                } else {
                    System.err.println("UCB1 ignores unknown customisation: " + input);
                }
            }
        }
    }
}
