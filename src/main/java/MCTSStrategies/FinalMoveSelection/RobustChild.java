package MCTSStrategies.FinalMoveSelection;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import main.collections.FVector;
import other.move.Move;
import other.state.State;
import search.mcts.MCTS;
import search.mcts.finalmoveselection.FinalMoveSelectionStrategy;
import search.mcts.nodes.BaseNode;
import utils.Value;

/**
 * Selects move corresponding to the most robust child (the highest visit count),
 * with an additional tie-breaker based on value estimates. If the MCTS
 * has a learned selection policy, that can be used as a second tie-breaker.
 * However, when a solved node is encountered this node will be used instead
 *
 * @author Dennis Soemers, made adjustments to also work with solver by Jelle Jansen
 */
public final class RobustChild implements FinalMoveSelectionStrategy
{

    //-------------------------------------------------------------------------

    /**
     * Selects the most robust child for the root node of the "actual" current game position
     *
     * @param mcts Ludii's mcts class
     * @param rootNode Node of the "actual" current game position
     * @return Move with the most visits
     */
    @Override
    public Move selectMove(final MCTS mcts, final BaseNode rootNode)
    {
        // Initialise all needed variables
        final List<Move> bestActions = new ArrayList<Move>();
        double bestActionValueEstimate = Double.NEGATIVE_INFINITY;
        float bestActionPolicyPrior = Float.NEGATIVE_INFINITY;
        final State rootState = rootNode.contextRef().state();
        final int moverAgent = rootState.playerToAgent(rootState.mover());
        int maxNumVisits = -1;

        // Extract the policy
        final FVector priorPolicy;
        if (mcts.learnedSelectionPolicy() == null)
            priorPolicy = null;
        else
            priorPolicy = rootNode.learnedSelectionPolicy();

        // Look for all children which child has the most number of visits with some tie-breakers
        final int numChildren = rootNode.numLegalMoves();
        for (int i = 0; i < numChildren; ++i)
        {
            final BaseNode child = rootNode.childForNthLegalMove(i);
            final int numVisits = child == null ? 0 : child.numVisits();
            final double childValueEstimate = child == null ? 0.0 : child.expectedScore(moverAgent);
            final float childPriorPolicy = priorPolicy == null ? -1.f : priorPolicy.get(i);
            boolean provenWin = child != null && child.totalScore(moverAgent) == Value.INF;

            // If the node is a proven win, return the winning node
            if (provenWin){
                return rootNode.nthLegalMove(i);
            }

            // If the move is more robust than the best known move save it
            if (numVisits > maxNumVisits)
            {
                maxNumVisits = numVisits;
                bestActions.clear();
                bestActionValueEstimate = childValueEstimate;
                bestActionPolicyPrior = childPriorPolicy;
                bestActions.add(rootNode.nthLegalMove(i));
            }
            // If the move is as good as the current move, check tie-breakers
            else if (numVisits == maxNumVisits)
            {
                if (childValueEstimate > bestActionValueEstimate)
                {
                    // Tie-breaker; prefer higher value estimates
                    bestActions.clear();
                    bestActionValueEstimate = childValueEstimate;
                    bestActionPolicyPrior = childPriorPolicy;
                    bestActions.add(rootNode.nthLegalMove(i));
                }
                else if (childValueEstimate == bestActionValueEstimate)
                {
                    // Tie for both num visits and also for estimated value; prefer higher prior policy
                    if (childPriorPolicy > bestActionPolicyPrior)
                    {
                        bestActions.clear();
                        bestActionValueEstimate = childValueEstimate;
                        bestActionPolicyPrior = childPriorPolicy;
                        bestActions.add(rootNode.nthLegalMove(i));
                    }
                    else if (childPriorPolicy == bestActionPolicyPrior)
                    {
                        // Tie for everything
                        bestActions.add(rootNode.nthLegalMove(i));
                    }
                }
            }
        }

        return bestActions.get(ThreadLocalRandom.current().nextInt(bestActions.size()));
    }

    //-------------------------------------------------------------------------

    @Override
    public void customise(final String[] inputs)
    {
        // Do nothing
    }

    //-------------------------------------------------------------------------

}
