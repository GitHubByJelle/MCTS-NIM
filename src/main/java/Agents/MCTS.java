package Agents;

import Evaluator.GameStateEvaluator;
import Evaluator.NeuralNetworkLeafEvaluator;
import Evaluator.ParallelNeuralNetworkLeafEvaluator;
import MCTSStrategies.Backpropagation.DynamicEarlyTerminationBackprop;
import MCTSStrategies.Backpropagation.FixedEarlyTerminationBackprop;
import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Node.implicitNode;
import MCTSStrategies.Node.implicitSolverNode;
import MCTSStrategies.Node.solverNode;
import MCTSStrategies.Playout.EpsilonGreedyPlayout;
import MCTSStrategies.Selection.ImplicitUCT;
import MCTSStrategies.Wrapper.EpsilonGreedySolvedSelectionWrapper;
import MCTSStrategies.Wrapper.TrainingPlayoutWrapper;
import MCTSStrategies.Wrapper.TrainingSelectionWrapper;
import MCTSStrategies.Wrapper.debugFinalSelectionWrapper;
import game.Game;
import other.RankUtils;
import other.context.Context;
import other.move.Move;
import other.state.State;
import other.trial.Trial;
import search.mcts.backpropagation.BackpropagationStrategy;
import search.mcts.backpropagation.MonteCarloBackprop;
import search.mcts.finalmoveselection.FinalMoveSelectionStrategy;
import search.mcts.nodes.BaseNode;
import search.mcts.nodes.OpenLoopNode;
import search.mcts.nodes.ScoreBoundsNode;
import search.mcts.nodes.StandardNode;
import search.mcts.playout.PlayoutStrategy;
import search.mcts.playout.RandomPlayout;
import search.mcts.selection.SelectionStrategy;
import search.mcts.selection.UCB1;
import utils.AIUtils;
import utils.Value;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * MCTS implementation from Ludii's Github, but adapted to work with the solver as proposed in
 * Winands, M. H., Björnsson, Y., & Saito, J. T. (2008, September). Monte-Carlo tree search solver. In International
 * Conference on Computers and Games (pp. 25-36). Springer, Berlin, Heidelberg.
 */
public class MCTS extends search.mcts.MCTS {

    //-------------------------------------------------------------------------

    /**
     * Indicates if the solver should be used
     */
    protected boolean useSolver = false;

    /**
     * Keeps track if the search needs to be terminated early
     */
    protected boolean stop = false;

    /**
     * GameStateEvaluator used to evaluate non-terminal leaf nodes
     */
    protected GameStateEvaluator leafEvaluator;

    /**
     * GameStateEvaluator used to evaluate terminal leaf nodes
     */
    protected GameStateEvaluator terminalStateEvaluator;

    /**
     * Indicates if the children should be evaluation batched (speeds up the iterations / sec when using NN)
     * Only works when using a Neural Network
     */
    protected boolean evaluateBatched = false;

    //-------------------------------------------------------------------------

    /**
     * Constructor of parallelized "default" MCTS algorithm with no inputs
     * UCB1, Random Play-outs, Monte Carlo Backpropagation and robust child on 6 threads.
     */
    public MCTS() {
        // Original
        super(new UCB1(), new RandomPlayout(-1),
                new MonteCarloBackprop(),
                new debugFinalSelectionWrapper(new RobustChild()));
        this.setNumThreads(6);
    }

    /**
     * Constructor requiring the architecture as input
     *
     * @param selectionStrategy          The used selection strategy
     * @param playoutStrategy            The used play-out strategy
     * @param backpropagationStrategy    The used backpropagation strategy
     * @param finalMoveSelectionStrategy The used final move selection strategy
     */
    public MCTS(SelectionStrategy selectionStrategy,
                PlayoutStrategy playoutStrategy,
                BackpropagationStrategy backpropagationStrategy,
                FinalMoveSelectionStrategy finalMoveSelectionStrategy) {
        // Original
        super(selectionStrategy, playoutStrategy, backpropagationStrategy,
                finalMoveSelectionStrategy);
    }

    /**
     * Selects and returns an action to play based on the MCTS architecture.
     *
     * @param game          Reference to the game we're playing.
     * @param context       Copy of the context containing the current state of the game
     * @param maxSeconds    Max number of seconds before a move should be selected.
     *                      Values less than 0 mean there is no time limit.
     * @param maxIterations Max number of iterations before a move should be selected.
     *                      Values less than 0 mean there is no iteration limit.
     * @param maxDepth      Max search depth before a move should be selected.
     *                      Values less than 0 mean there is no search depth limit.
     * @return Preferred move.
     */
    @Override
    public Move selectAction
    (
            final Game game,
            final Context context,
            final double maxSeconds,
            final int maxIterations,
            final int maxDepth
    ) {
        final long startTime = System.currentTimeMillis();
        long stopTime = (maxSeconds > 0.0) ? startTime + (long) (maxSeconds * 1000) : Long.MAX_VALUE;
        final int maxIts = (maxIterations >= 0) ? maxIterations : Integer.MAX_VALUE;

        while (numThreadsBusy.get() != 0 && System.currentTimeMillis() < Math.min(stopTime, startTime + 1000L)) {
            // Give threads in thread pool some more time to clean up after themselves from previous iteration
        }

        // We'll assume all threads are really done now and just reset to 0
        numThreadsBusy.set(0);

        final AtomicInteger numIterations = new AtomicInteger();

        // Find or create root node
        if (treeReuse && rootNode != null) {
            // Want to reuse part of existing search tree

            // Need to traverse parts of old tree corresponding to
            // actions played in the real game
            final List<Move> actionHistory = context.trial().generateCompleteMovesList();
            int offsetActionToTraverse = actionHistory.size() - lastActionHistorySize;

            if (offsetActionToTraverse < 0) {
                // Something strange happened, probably forgot to call
                // initAI() for a newly-started game. Won't be a good
                // idea to reuse tree anyway
                rootNode = null;
            }

            while (offsetActionToTraverse > 0) {
                final Move move = actionHistory.get(actionHistory.size() - offsetActionToTraverse);
                rootNode = rootNode.findChildForMove(move);

                if (rootNode == null) {
                    // Didn't have a node in tree corresponding to action
                    // played, so can't reuse tree
                    break;
                }

                --offsetActionToTraverse;
            }
        }

        if (rootNode == null || !treeReuse) {
            // Need to create a fresh root
            rootNode = createNode(this, null, null, null, context);
            //System.out.println("NO TREE REUSE");
        } else {
            //System.out.println("successful tree reuse");

            // We're reusing a part of previous search tree
            // Clean up unused parts of search tree from memory
            rootNode.setParent(null);

            // TODO in nondeterministic games + OpenLoop MCTS, we'll want to
            // decay statistics gathered in the entire subtree here
        }

        if (globalActionStats != null) {
            // Decay global action statistics
            final Set<Entry<MoveKey, ActionStatistics>> entries = globalActionStats.entrySet();
            final Iterator<Entry<MoveKey, ActionStatistics>> it = entries.iterator();

            while (it.hasNext()) {
                final Entry<MoveKey, ActionStatistics> entry = it.next();
                final ActionStatistics stats = entry.getValue();
                stats.visitCount *= globalActionDecayFactor;

                if (stats.visitCount < 1.0)
                    it.remove();
                else
                    stats.accumulatedScore *= globalActionDecayFactor;
            }
        }

        if (globalNGramActionStats != null) {
            // Decay global N-gram action statistics
            final Set<Entry<NGramMoveKey, ActionStatistics>> entries = globalNGramActionStats.entrySet();
            final Iterator<Entry<NGramMoveKey, ActionStatistics>> it = entries.iterator();

            while (it.hasNext()) {
                final Entry<NGramMoveKey, ActionStatistics> entry = it.next();
                final ActionStatistics stats = entry.getValue();
                stats.visitCount *= globalActionDecayFactor;

                if (stats.visitCount < 1.0)
                    it.remove();
                else
                    stats.accumulatedScore *= globalActionDecayFactor;
            }
        }

        if (heuristicStats != null) {
            // Clear all heuristic stats
            for (int p = 1; p < heuristicStats.length; ++p) {
                heuristicStats[p].init(0, 0.0, 0.0);
            }
        }

        rootNode.rootInit(context);

        if (rootNode.numLegalMoves() == 1) {
            // play faster if we only have one move available anyway
            if (autoPlaySeconds >= 0.0 && autoPlaySeconds < maxSeconds)
                stopTime = startTime + (long) (autoPlaySeconds * 1000);
        }

        lastActionHistorySize = context.trial().numMoves();

        lastNumPlayoutActions = 0;    // TODO if this variable actually becomes important, may want to make it Atomic

        // Set the stop variable for early stop to false, since no search has been done
        this.setStop(false);

        // Store this in a separate variable because threading weirdness sometimes sets the class variable to null
        // even though some threads here still want to do something with it.
        final BaseNode rootThisCall = rootNode;

        // Select the mover for the current root
        final int mover = context.state().playerToAgent(context.state().mover());

        // For each thread, queue up a job
        final CountDownLatch latch = new CountDownLatch(numThreads);
        final long finalStopTime = stopTime;    // Need this to be final for use in inner lambda
        for (int thread = 0; thread < numThreads; ++thread) {
            threadPool.submit
                    (
                            () ->
                            {
                                try {
                                    numThreadsBusy.incrementAndGet();

                                    // Search until we have to stop
                                    while (!this.earlyStop(rootThisCall, mover) &&
                                            numIterations.get() < maxIts && System.currentTimeMillis() < finalStopTime
                                            && !wantsInterrupt) {
                                        /*********************
                                         Selection Phase
                                         *********************/
                                        BaseNode current = rootThisCall;
                                        current.addVirtualVisit();
                                        current.startNewIteration(context);

                                        Context playoutContext = null;

                                        while (current.contextRef().trial().status() == null) {
                                            BaseNode prevNode = current;
                                            prevNode.getLock().lock();

                                            try {
                                                // If current node is proven, stop selection
                                                if (this.useSolver && current.isValueProven(mover)) {
                                                    break;
                                                }

                                                // Else perform selection
                                                final int selectedIdx = selectionStrategy.select(this, current);
                                                BaseNode nextNode = current.childForNthLegalMove(selectedIdx);

                                                final Context newContext = current.traverse(selectedIdx);

                                                if (nextNode == null) {
                                                    /*********************
                                                     Expand
                                                     *********************/
                                                    nextNode =
                                                            createNode
                                                                    (
                                                                            this,
                                                                            current,
                                                                            newContext.trial().lastMove(),
                                                                            current.nthLegalMove(selectedIdx),
                                                                            newContext
                                                                    );

                                                    current.addChild(nextNode, selectedIdx);
                                                    current = nextNode;
                                                    current.addVirtualVisit();
                                                    current.updateContextRef();

                                                    if ((expansionFlags & HEURISTIC_INIT) != 0) {
                                                        assert (heuristicFunction != null);
                                                        nextNode.setHeuristicValueEstimates
                                                                (
                                                                        AIUtils.heuristicValueEstimates(nextNode.playoutContext(), heuristicFunction)
                                                                );
                                                    }

                                                    playoutContext = current.playoutContext();

                                                    break;    // stop Selection phase
                                                }

                                                current = nextNode;
                                                current.addVirtualVisit();
                                                current.updateContextRef();
                                            } catch (final ArrayIndexOutOfBoundsException e) {
                                                System.err.println(describeMCTS());
                                                throw e;
                                            } finally {
                                                prevNode.getLock().unlock();
                                            }
                                        }


                                        // If value is proven, update game theoretical values
                                        if (this.useSolver &&
                                                (current.isValueProven(
                                                        current.contextRef().state().playerToAgent(
                                                                current.contextRef().state().mover())))) {

                                            /********************************
                                             Solved position found
                                             ********************************/
                                            boolean updateGRAVE = (this.backpropFlags & 1) != 0;
                                            boolean updateGlobalActionStats = (this.backpropFlags & 2) != 0;
                                            boolean updateGlobalNGramActionStats = (this.backpropFlags & 4) != 0;
                                            List<search.mcts.MCTS.MoveKey> moveKeysAMAF = new ArrayList();
                                            int movesIdxAMAF = current.contextRef().trial().numMoves() - 1;
                                            Iterator<Move> reverseMovesIterator = current.contextRef().trial().reverseMoveIterator();

//                                            System.out.println("Current solved = " + current.isValueProven(mover) + ")");
//                                            System.out.println("List size: " + moveKeysAMAF.size());
//                                            System.out.println("Root depth: " + (rootThisCall.contextRef().trial().numMoves()-1) +
//                                                    " (solved = " + rootThisCall.isValueProven(mover) + ")");
//                                            System.out.println("Move idx: " + movesIdxAMAF);
//                                            System.out.println("--");

                                            current.updateGameTheoreticalValues(updateGRAVE, updateGlobalActionStats,
                                                    moveKeysAMAF, movesIdxAMAF, reverseMovesIterator,
                                                    current.totalScores());

//                                            System.out.println("Current solved = " + current.isValueProven(mover) + ")");
//                                            System.out.println("List size: " + moveKeysAMAF.size());
//                                            System.out.println("Root depth: " + (rootThisCall.contextRef().trial().numMoves()-1) +
//                                                    " (solved = " + rootThisCall.isValueProven(mover) + ")");
//                                            System.out.println("Move idx: " + movesIdxAMAF);
//                                            System.out.println();

                                            double[] tempUtil = new double[3];
                                            for (int i = 1; i < 3; i++) {
                                                tempUtil[i] = current.totalScore(i) / Value.INF;
                                            }

                                            this.backpropagationStrategy.updateGlobalActionStats(this, updateGlobalActionStats,
                                                    updateGlobalNGramActionStats, moveKeysAMAF, current.contextRef(),
                                                    tempUtil);
                                        }
                                        // Else use backpropagation strategy of Ludii
                                        else {
                                            Trial endTrial = current.contextRef().trial();
                                            int numPlayoutActions = 0;

                                            if (!endTrial.over() && playoutValueWeight > 0.0) {
                                                // Did not reach a terminal game state yet

                                                /********************************
                                                 Play-out
                                                 ********************************/

                                                final int numActionsBeforePlayout = current.contextRef().trial().numMoves();

                                                endTrial = playoutStrategy.runPlayout(this, playoutContext);
                                                numPlayoutActions = (endTrial.numMoves() - numActionsBeforePlayout);

                                                lastNumPlayoutActions +=
                                                        (playoutContext.trial().numMoves() - numActionsBeforePlayout);
                                            } else {
                                                // Reached a terminal game state
                                                playoutContext = current.contextRef();
                                            }

                                            /***************************
                                             Backpropagation Phase
                                             ***************************/
                                            final double[] outcome = RankUtils.agentUtilities(playoutContext);
                                            backpropagationStrategy.update(this, current, playoutContext, outcome, numPlayoutActions);
                                        }

                                        numIterations.incrementAndGet();
                                    }

                                    rootThisCall.cleanThreadLocals();
                                } catch (final Exception e) {
                                    System.err.println("MCTS error in game: " + context.game().name());
                                    e.printStackTrace();    // Need to do this here since we don't retrieve runnable's Future result
                                } finally {
                                    numThreadsBusy.decrementAndGet();
                                    latch.countDown();
                                }
                            }
                    );
        }

        try {
            latch.await(stopTime - startTime + 2000L, TimeUnit.MILLISECONDS);
        } catch (final InterruptedException e) {
            e.printStackTrace();
        }

        lastNumMctsIterations = numIterations.get();

        final Move returnMove = finalMoveSelectionStrategy.selectMove(this, rootThisCall);
        int playedChildIdx = -1;

        if (!wantsInterrupt) {
            int moveVisits = -1;

            for (int i = 0; i < rootThisCall.numLegalMoves(); ++i) {
                final BaseNode child = rootThisCall.childForNthLegalMove(i);

                if (child != null) {
                    if (rootThisCall.nthLegalMove(i).equals(returnMove)) {
                        final State state = rootThisCall.deterministicContextRef().state();
                        final int moverAgent = state.playerToAgent(state.mover());
                        moveVisits = child.numVisits();
                        lastReturnedMoveValueEst = child.expectedScore(moverAgent);
                        playedChildIdx = i;

                        break;
                    }
                }
            }

            final int numRootIts = rootThisCall.numVisits();

            analysisReport =
                    friendlyName +
                            " made move after " +
                            numRootIts +
                            " iterations (selected child visits = " +
                            moveVisits +
                            ", value = " +
                            lastReturnedMoveValueEst +
                            ").";
        } else {
            analysisReport = null;
        }

        // We can already try to clean up a bit of memory here
        // NOTE: from this point on we have to use rootNode instead of rootThisCall again!
        if (!preserveRootNode) {
            if (!treeReuse) {
                rootNode = null;    // clean up entire search tree
            } else if (!wantsInterrupt)    // only clean up if we didn't pause the AI / interrupt it
            {
                if (playedChildIdx >= 0)
                    rootNode = rootThisCall.childForNthLegalMove(playedChildIdx);
                else
                    rootNode = null;

                if (rootNode != null) {
                    rootNode.setParent(null);
                    ++lastActionHistorySize;
                }
            }
        }

        return returnMove;
    }

    /**
     * Creates a node belonging to the used selection strategy
     *
     * @param mcts                    Ludii's MCTS algorithm
     * @param parent                  Parent node of newly created nodes
     * @param parentMove              Move from parent node to newly created node
     * @param parentMoveWithoutConseq Move from parent node to newly created node
     * @param context                 Ludii's context representing the game position of the new node
     * @return New child node
     */
    @Override
    protected BaseNode createNode(search.mcts.MCTS mcts, BaseNode parent, Move parentMove, Move parentMoveWithoutConseq, Context context) {
        if ((this.currentGameFlags & 64L) != 0L && !this.wantsCheatRNG()) {
            return new OpenLoopNode(mcts, parent, parentMove, parentMoveWithoutConseq, context.game());
        } else {
            if (this.selectionStrategy instanceof ImplicitUCT ||
                    (this.selectionStrategy instanceof EpsilonGreedySolvedSelectionWrapper &&
                            ((EpsilonGreedySolvedSelectionWrapper) this.selectionStrategy).selectionStrategy
                                    instanceof ImplicitUCT)) {
                if (this.useSolver) {
//                    if (this.useTT){
//                        return new implicitSolverNodeTT(mcts, parent, parentMove, parentMoveWithoutConseq, context);
//                    }

                    return new implicitSolverNode(mcts, parent, parentMove, parentMoveWithoutConseq, context,
                            this.leafEvaluator, this.terminalStateEvaluator, this.evaluateBatched);
                } else {
//                    if (this.useTT){
//                        return new implicitNodeTT(mcts, parent, parentMove, parentMoveWithoutConseq, context);
//                    }

                    return new implicitNode(mcts, parent, parentMove, parentMoveWithoutConseq, context,
                            this.leafEvaluator, this.terminalStateEvaluator, this.evaluateBatched);
                }
            } else if (this.useScoreBounds) {
                return new ScoreBoundsNode(mcts, parent, parentMove, parentMoveWithoutConseq, context);
            } else if (this.useSolver) {
                return new solverNode(mcts, parent, parentMove, parentMoveWithoutConseq, context);
            }

            return (BaseNode) (new StandardNode(mcts, parent, parentMove, parentMoveWithoutConseq, context));
        }
    }

    /**
     * Perform desired initialisation before starting to play a game
     *
     * @param game     The game that we'll be playing
     * @param playerID The player ID for the AI in this game
     */
    public void initAI(Game game, int playerID) {
        this.initParent(game, playerID);
    }

    /**
     * Initialises the game as done by Ludii's MCTS algorithm (needed for inherited classes)
     *
     * @param game     Ludii's game class
     * @param playerID ID of player
     */
    public void initParent(Game game, int playerID) {
        super.initAI(game, playerID);
    }

    /**
     * Sets the GameStateEvaluator to all objects which require the GameStateEvaluator
     *
     * @param leafEvaluator GameStateEvaluator used to evaluate non-terminal leaf nodes
     * @param game          Ludii's game class
     */
    protected void setLeafEvaluator(GameStateEvaluator leafEvaluator, Game game) {
        // Saves the leaf evaluator to the MCTS bot
        this.leafEvaluator = leafEvaluator;

        // Checks if evaluations should be performed batched (to increase iterations / sec)
        if (leafEvaluator instanceof NeuralNetworkLeafEvaluator) {
            this.evaluateBatched = true;
        }

        // Sets the evaluation function to the play-out strategy if needed
        if (this.playoutStrategy instanceof TrainingPlayoutWrapper) {
            if (leafEvaluator instanceof NeuralNetworkLeafEvaluator) {
                ((TrainingPlayoutWrapper) this.playoutStrategy).createBatchedMoveSelector();
            }

            ((TrainingPlayoutWrapper) this.playoutStrategy).setLeafEvaluator(leafEvaluator);
        } else if (this.playoutStrategy instanceof EpsilonGreedyPlayout) {
            if (leafEvaluator instanceof NeuralNetworkLeafEvaluator) {
                ((EpsilonGreedyPlayout) this.playoutStrategy).createBatchedMoveSelector();
            }

            ((EpsilonGreedyPlayout) this.playoutStrategy).setLeafEvaluator(leafEvaluator);
        }

        // Sets the evaluation function to the backpropagation strategy if needed
        if (this.backpropagationStrategy instanceof FixedEarlyTerminationBackprop) {
            ((FixedEarlyTerminationBackprop) this.backpropagationStrategy).setLeafEvaluator(leafEvaluator);
        } else if (this.backpropagationStrategy instanceof DynamicEarlyTerminationBackprop) {
            ((DynamicEarlyTerminationBackprop) this.backpropagationStrategy).setLeafEvaluator(leafEvaluator);
        }

        // Sets the evaluation function to the final move selection strategy if needed
        if (finalMoveSelectionStrategy instanceof TrainingSelectionWrapper) {
            ((TrainingSelectionWrapper) this.finalMoveSelectionStrategy).createLeafEvaluator(game);
        }
    }

    /**
     * Sets the GameStateEvaluator to all objects which require the GameStateEvaluator
     *
     * @param terminalStateEvaluator GameStateEvaluator used to evaluate terminal nodes
     */
    protected void setTerminalStateEvaluator(GameStateEvaluator terminalStateEvaluator) {
        // Saves the terminal state evaluator to the MCTS bot
        this.terminalStateEvaluator = terminalStateEvaluator;

        // Sets the evaluation function to the play-out strategy if needed
        if (this.playoutStrategy instanceof TrainingPlayoutWrapper) {
            ((TrainingPlayoutWrapper) this.playoutStrategy).setTerminalStateEvaluator(terminalStateEvaluator);
        } else if (this.playoutStrategy instanceof EpsilonGreedyPlayout) {
            ((EpsilonGreedyPlayout) this.playoutStrategy).setTerminalStateEvaluator(terminalStateEvaluator);
        }
    }

    /**
     * Setter for the solver
     *
     * @param useSolver true if solver should be used
     */
    public void setUseSolver(boolean useSolver) {
        this.useSolver = useSolver;
    }

    /**
     * Setter for the stop
     *
     * @param stop true if the search algorithm should be stopped early
     */
    public void setStop(boolean stop) {
        this.stop = stop;
    }

    /**
     * Indicates if the algorithm should stop early by checking if the rootnode is solved.
     *
     * @param rootThisCall Root node of current call
     * @param mover        ID of the player to move
     * @return true if the search algorithm needs to stop
     */
    protected boolean earlyStop(final BaseNode rootThisCall, final int mover) {
        if (this.useSolver) {
            return rootThisCall.isValueProven(mover);
        }

        return this.stop;
    }

    /**
     * Closes the AI and GameStateEvaluators correctly after being used!
     * Needs to be executed when using this function, otherwise to many parallel-inference threads will be created
     */
    @Override
    public void closeAI() {
        super.closeAI();

        if (this.leafEvaluator instanceof ParallelNeuralNetworkLeafEvaluator) {
            ((ParallelNeuralNetworkLeafEvaluator) this.leafEvaluator).close();
        }
    }
}