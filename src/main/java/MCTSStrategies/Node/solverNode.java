package MCTSStrategies.Node;

import other.RankUtils;
import other.context.Context;
import other.move.Move;
import search.mcts.MCTS;
import search.mcts.nodes.BaseNode;
import search.mcts.nodes.DeterministicNode;
import utils.Value;

import java.util.Iterator;
import java.util.List;

/**
 * Node class which allows the adapted Ludii MCTS class to be used with an solver.
 */
public class solverNode extends DeterministicNode {

    /**
     * Constructor for the solver node
     *
     * @param mcts                    Ludii's MCTS class
     * @param parent                  Parent node of current node
     * @param parentMove              Node from parent to current node
     * @param parentMoveWithoutConseq Node from parent to current node
     * @param context                 Ludii's context class representating the game state
     */
    public solverNode(MCTS mcts, BaseNode parent, Move parentMove, Move parentMoveWithoutConseq, Context context) {
        super(mcts, parent, parentMove, parentMoveWithoutConseq, context);

        // Check if current state is winning
        if (context.trial().over()) {
            final double[] utilities = RankUtils.agentUtilities(context);
            for (int p = 1; p < utilities.length; ++p) {
                utilities[p] *= Value.INF;
            }

            // Update total score
            synchronized (this) {
                for (int p = 1; p < utilities.length; ++p) {
                    this.totalScores[p] = utilities[p];
                }
            }
        } else {
            // Check if any unexpanded child (all legal moves) leads to a win for the player to move
            final int numLegalMoves = this.numLegalMoves();
            for (int i = 0; i < numLegalMoves; i++) {
                Context contextCopy = new Context(this.contextRef());
                contextCopy.game().apply(contextCopy, this.legalMoves[i]);

                // If it is terminal and a win, it's solved as well
                if (contextCopy.trial().over()) {
                    final int mover = context.state().playerToAgent(context.state().mover());
                    final double[] utilities = RankUtils.utilities(contextCopy);
                    if (utilities[mover] == 1) {
                        for (int p = 1; p < utilities.length; ++p) {
                            utilities[p] *= Value.INF;
                        }

                        // Update total score
                        synchronized (this) {
                            for (int p = 1; p < utilities.length; ++p) {
                                this.totalScores[p] = utilities[p];
                            }
                        }

                        // Add child (otherwise won't be seen by final move selection for root node,
                        // since child will be null)
                        this.addChild(new solverNode(mcts, this, contextCopy.trial().lastMove(),
                                this.nthLegalMove(i), contextCopy), i);

                        break;
                    }
                }
            }
        }
    }

    /**
     * Updates the game theoretical values (proven win or loss) based on the backpropagation described in
     * Winands, M. H., Björnsson, Y., & Saito, J. T. (2008, September). Monte-Carlo tree search solver. In
     * International Conference on Computers and Games (pp. 25-36). Springer, Berlin, Heidelberg.
     *
     * @param updateGRAVE             Indicates if GRAVE needs to be updates (as used by Ludii)
     * @param updateGlobalActionStats Indicates if the GlobalActionStats needs to be updates (as used by Ludii)
     * @param moveKeysAMAF            List of the current moveKeysAMAF to which more data will be added (as used by Ludii)
     * @param movesIdxAMAF            List of the current moveIdxAMAF to which more data will be added (as used by Ludii)
     * @param reverseMovesIterator    Iterator of moves played (as used by Ludii)
     * @param utilities               Score for evaluated game position or Game Theoretical values (9999999 for proven win,
     *                                -9999999 for proven loss w.r.t. player one)
     */
    public void updateGameTheoreticalValues(boolean updateGRAVE, boolean updateGlobalActionStats,
                                            List<MCTS.MoveKey> moveKeysAMAF, int movesIdxAMAF,
                                            Iterator<Move> reverseMovesIterator, final double[] utilities) {
        final int mover = this.context.state().playerToAgent(this.context.state().mover());

        // If proven win, just update this node
        if (utilities[mover] == Value.INF) {
            synchronized (this) {
                this.update(utilities);

                if (updateGRAVE) {
                    this.updateRave(moveKeysAMAF, utilities, mover);
                }
            }

            if (updateGRAVE || updateGlobalActionStats) {
                // we're going up one level, so also one more move to count as AMAF-move
                if (movesIdxAMAF >= 0) {
                    moveKeysAMAF.add(new MCTS.MoveKey(reverseMovesIterator.next(), movesIdxAMAF));
                    --movesIdxAMAF;
                }
            }

            // Update parent
            if (this.parent != null) {
                this.parent.updateGameTheoreticalValues(updateGRAVE, updateGlobalActionStats,
                        moveKeysAMAF, movesIdxAMAF, reverseMovesIterator, utilities);
            }
        } else {
            // If proven loss, check if all children are a proven loss, else return normal loss (value of -1)
            ifstatement:
            if (utilities[mover] == -Value.INF) {
                for (DeterministicNode child : this.children) {
                    if (child == null || child.totalScore(mover) != -Value.INF) {
                        double[] tempUtil = new double[3];
                        tempUtil[mover] = -1;
                        tempUtil[mover == 1 ? 2 : 1] = 1;

                        BaseNode current = this;
                        while (current != null) {
                            synchronized (current) {
                                current.update(tempUtil);

                                if (updateGRAVE) {
                                    ((solverNode) current).updateRave(moveKeysAMAF, tempUtil, mover);
                                }
                            }

                            if (updateGRAVE || updateGlobalActionStats) {
                                // we're going up one level, so also one more move to count as AMAF-move
                                if (movesIdxAMAF >= 0) {
                                    moveKeysAMAF.add(new MCTS.MoveKey(reverseMovesIterator.next(), movesIdxAMAF));
                                    --movesIdxAMAF;
                                }
                            }

                            current = current.parent();
                        }

                        break ifstatement;
                    }
                }

                synchronized (this) {
                    this.update(utilities);

                    if (updateGRAVE) {
                        this.updateRave(moveKeysAMAF, utilities, mover);
                    }
                }

                if (updateGRAVE || updateGlobalActionStats) {
                    // we're going up one level, so also one more move to count as AMAF-move
                    if (movesIdxAMAF >= 0) {
                        moveKeysAMAF.add(new MCTS.MoveKey(reverseMovesIterator.next(), movesIdxAMAF));
                        --movesIdxAMAF;
                    }
                }

                // Update parent
                if (this.parent != null) {
                    this.parent.updateGameTheoreticalValues(updateGRAVE, updateGlobalActionStats,
                            moveKeysAMAF, movesIdxAMAF, reverseMovesIterator, utilities);
                }
            }
        }
    }

    /**
     * Checks if current node is proven
     *
     * @param agent Id of agent to move
     * @return Indicates if current node is proven
     */
    public boolean isValueProven(int agent) {
        return Math.abs(this.totalScores[agent]) == Value.INF;
    }

    /**
     * Returns the expected score of the node. If the node is solved, the game theoretical value will be returned,
     * else the "normal" expected score will be returned (see BaseNode)
     *
     * @param agent Id of agent to move
     * @return Expected score
     */
    public double expectedScore(int agent) {
        return this.isValueProven(agent) ? this.totalScore(agent) : super.expectedScore(agent);
    }

    /**
     * Updates the RAVE values for the solved nodes. If a game theoretical value needs to be returned, the accumulated
     * score will be INF. Else the normal score will be added.
     *
     * @param moveKeysAMAF List of the current moveKeysAMAF to which more data will be added (as used by Ludii)
     * @param utilities    Score for evaluated game position or Game Theoretical values (9999999 for proven win,
     *                     -9999999 for proven loss w.r.t. player one)
     * @param mover        Id of agent to move
     */
    public void updateRave(List<MCTS.MoveKey> moveKeysAMAF, double[] utilities, final int mover) {
        for (final MCTS.MoveKey moveKey : moveKeysAMAF) {
            // Get entry and add visit count
            final NodeStatistics graveStats = this.getOrCreateGraveStatsEntry(moveKey);
            graveStats.visitCount += 1;

            // If a game theoretical value needs to be returned, the accumulated score will be INF.
            // Else the normal score will be added.
            if (Math.abs(utilities[mover]) == Value.INF) {
                graveStats.accumulatedScore = utilities[mover];
            } else {
                graveStats.accumulatedScore += utilities[mover];
            }
        }
    }

    /**
     * Updates the node based on the given utilities without exceeding the game theoretical values
     *
     * @param utilities Score for evaluated game position or Game Theoretical values (9999999 for proven win,
     *                  -9999999 for proven loss w.r.t. player one)
     */
    public void update(double[] utilities) {
        // Checks if the score isn't to high
        if (Math.abs(this.totalScore(1)) > Value.INF) {
            throw new RuntimeException("The total score exceeded the INF value. Debug the code.");
        }

        // If both agents are proven, only update the visits
        // Else if the utilities contain a game theoretical win for any of the players, set the value to the
        //         Game Theoretical Value.
        // Else Update the node normally
        if (isValueProven(1) || isValueProven(2)) {
            this.numVisits++;
            numVirtualVisits.decrementAndGet();
        } else if (Math.abs(utilities[1]) == Value.INF || Math.abs(utilities[2]) == Value.INF) {
            this.numVisits++;
            for (int p = 1; p < totalScores.length; ++p) {
                totalScores[p] = utilities[p];
                sumSquaredScores[p] = utilities[p] * utilities[p];
            }
            numVirtualVisits.decrementAndGet();
        } else {
            super.update(utilities);
        }
    }
}
