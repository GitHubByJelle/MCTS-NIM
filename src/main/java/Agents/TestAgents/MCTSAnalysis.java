package Agents;

import MCTSStrategies.FinalMoveSelection.RobustChild;
import MCTSStrategies.Node.implicitNode;
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
import search.mcts.playout.PlayoutStrategy;
import search.mcts.playout.RandomPlayout;
import search.mcts.selection.SelectionStrategy;
import search.mcts.selection.UCB1;
import utils.AIUtils;
import utils.Value;

import java.text.DecimalFormat;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Adapted MCTS implementation from Ludii's Github which analyses all the search performed by the
 * search algorithm every iteration. Please note, the analyzer requires to also
 * store the parameters of exploration and influence to the bot.
 */
public class MCTSAnalysis extends Agents.MCTS {

    //-------------------------------------------------------------------------

    /** The initial node of the game */
    private BaseNode initialGameStateRoot;

    /** Exploration constant used */
    protected double explorationConstant;

    /** Influence on estimated values used */
    protected double influenceEstimatedValues;

    //-------------------------------------------------------------------------

    /**
     * Constructor of parallelized "default" MCTS algorithm with no inputs
     * UCB1, Random Play-outs, Monte Carlo Backpropagation and robust child on 6 threads.
     */
    public MCTSAnalysis() {
        // Original
        super(new UCB1(), new RandomPlayout(-1),
                new MonteCarloBackprop(), new RobustChild());
        this.setNumThreads(6);

        this.explorationConstant = Math.sqrt(2);
    }

    /**
     * Constructor requiring the architecture as input
     *
     * @param selectionStrategy The used selection strategy
     * @param playoutStrategy The used play-out strategy
     * @param backpropagationStrategy The used backpropagation strategy
     * @param finalMoveSelectionStrategy The used final move selection strategy
     */
    public MCTSAnalysis(SelectionStrategy selectionStrategy,
                        PlayoutStrategy playoutStrategy,
                        BackpropagationStrategy backpropagationStrategy,
                        FinalMoveSelectionStrategy finalMoveSelectionStrategy) {
        // Original
        super(selectionStrategy, playoutStrategy, backpropagationStrategy,
                finalMoveSelectionStrategy);
    }

    /**
     * Selects and returns an action to play based on the MCTS architecture, while also printing
     * an analysis of the search performed
     *
     * @param game Reference to the game we're playing.
     * @param context Copy of the context containing the current state of the game
     * @param maxSeconds Max number of seconds before a move should be selected.
     * Values less than 0 mean there is no time limit.
     * @param maxIterations Max number of iterations before a move should be selected.
     * Values less than 0 mean there is no iteration limit.
     * @param maxDepth Max search depth before a move should be selected.
     * Values less than 0 mean there is no search depth limit.
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
            )
    {
        final long startTime = System.currentTimeMillis();
        long stopTime = (maxSeconds > 0.0) ? startTime + (long) (maxSeconds * 1000) : Long.MAX_VALUE;
        final int maxIts = (maxIterations >= 0) ? maxIterations : Integer.MAX_VALUE;

        while (numThreadsBusy.get() != 0 && System.currentTimeMillis() < Math.min(stopTime, startTime + 1000L))
        {
            // Give threads in thread pool some more time to clean up after themselves from previous iteration
        }

        // We'll assume all threads are really done now and just reset to 0
        numThreadsBusy.set(0);

        final AtomicInteger numIterations = new AtomicInteger();

        // Find or create root node
        if (treeReuse && rootNode != null)
        {
            // Want to reuse part of existing search tree

            // Need to traverse parts of old tree corresponding to
            // actions played in the real game
            final List<Move> actionHistory = context.trial().generateCompleteMovesList();
            int offsetActionToTraverse = actionHistory.size() - lastActionHistorySize;

            if (offsetActionToTraverse < 0)
            {
                // Something strange happened, probably forgot to call
                // initAI() for a newly-started game. Won't be a good
                // idea to reuse tree anyway
                rootNode = null;
            }

            while (offsetActionToTraverse > 0)
            {
                final Move move = actionHistory.get(actionHistory.size() - offsetActionToTraverse);
                rootNode = rootNode.findChildForMove(move);

                if (rootNode == null)
                {
                    // Didn't have a node in tree corresponding to action
                    // played, so can't reuse tree
                    break;
                }

                --offsetActionToTraverse;
            }
        }

        if (rootNode == null || !treeReuse)
        {
            rootNode = createNode(this, null, null, null, context);
            initialGameStateRoot = null;
            //System.out.println("NO TREE REUSE");
        }
        else
        {
            //System.out.println("successful tree reuse");

            // We're reusing a part of previous search tree
            // Clean up unused parts of search tree from memory
            rootNode.setParent(null);

            // TODO in nondeterministic games + OpenLoop MCTS, we'll want to
            // decay statistics gathered in the entire subtree here
        }

        if (globalActionStats != null)
        {
            // Decay global action statistics
            final Set<Entry<MoveKey, ActionStatistics>> entries = globalActionStats.entrySet();
            final Iterator<Entry<MoveKey, ActionStatistics>> it = entries.iterator();

            while (it.hasNext())
            {
                final Entry<MoveKey, ActionStatistics> entry = it.next();
                final ActionStatistics stats = entry.getValue();
                stats.visitCount *= globalActionDecayFactor;

                if (stats.visitCount < 1.0)
                    it.remove();
                else
                    stats.accumulatedScore *= globalActionDecayFactor;
            }
        }

        if (globalNGramActionStats != null)
        {
            // Decay global N-gram action statistics
            final Set<Entry<NGramMoveKey, ActionStatistics>> entries = globalNGramActionStats.entrySet();
            final Iterator<Entry<NGramMoveKey, ActionStatistics>> it = entries.iterator();

            while (it.hasNext())
            {
                final Entry<NGramMoveKey, ActionStatistics> entry = it.next();
                final ActionStatistics stats = entry.getValue();
                stats.visitCount *= globalActionDecayFactor;

                if (stats.visitCount < 1.0)
                    it.remove();
                else
                    stats.accumulatedScore *= globalActionDecayFactor;
            }
        }

        if (heuristicStats != null)
        {
            // Clear all heuristic stats
            for (int p = 1; p < heuristicStats.length; ++p)
            {
                heuristicStats[p].init(0, 0.0, 0.0);
            }
        }

        rootNode.rootInit(context);

        if (rootNode.numLegalMoves() == 1)
        {
            // play faster if we only have one move available anyway
            if (autoPlaySeconds >= 0.0 && autoPlaySeconds < maxSeconds)
                stopTime = startTime + (long) (autoPlaySeconds * 1000);
        }

        lastActionHistorySize = context.trial().numMoves();

        lastNumPlayoutActions = 0;	// TODO if this variable actually becomes important, may want to make it Atomic

        // Store this in a separate variable because threading weirdness sometimes sets the class variable to null
        // even though some threads here still want to do something with it.
        final BaseNode rootThisCall = rootNode;

        // Select the mover for the current root
        final int mover = context.state().playerToAgent(context.state().mover());

        // For each thread, queue up a job
        final CountDownLatch latch = new CountDownLatch(numThreads);
        final long finalStopTime = stopTime;	// Need this to be final for use in inner lambda
        for (int thread = 0; thread < numThreads; ++thread)
        {
            threadPool.submit
                    (
                            () ->
                            {
                                try
                                {
                                    numThreadsBusy.incrementAndGet();

                                    // Search until we have to stop
                                    while (!this.earlyStop(rootThisCall, mover) &&
                                            numIterations.get() < maxIts && System.currentTimeMillis() < finalStopTime
                                            && !wantsInterrupt)
                                    {
                                        /*********************
                                         Selection Phase
                                         *********************/
                                        BaseNode current = rootThisCall;
                                        current.addVirtualVisit();
                                        current.startNewIteration(context);

                                        Context playoutContext = null;

                                        while (current.contextRef().trial().status() == null)
                                        {
                                            BaseNode prevNode = current;
                                            prevNode.getLock().lock();

                                            try
                                            {
                                                // If current node is proven, stop selection
                                                if (this.useSolver && current.isValueProven(mover)){
                                                    break;
                                                }

                                                // Else perform selection
                                                final int selectedIdx = selectionStrategy.select(this, current);
                                                BaseNode nextNode = current.childForNthLegalMove(selectedIdx);

                                                final Context newContext = current.traverse(selectedIdx);

                                                if (nextNode == null)
                                                {
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

                                                    if ((expansionFlags & HEURISTIC_INIT) != 0)
                                                    {
                                                        assert (heuristicFunction != null);
                                                        nextNode.setHeuristicValueEstimates
                                                                (
                                                                        AIUtils.heuristicValueEstimates(nextNode.playoutContext(), heuristicFunction)
                                                                );
                                                    }

                                                    playoutContext = current.playoutContext();

                                                    break;	// stop Selection phase
                                                }

                                                current = nextNode;
                                                current.addVirtualVisit();
                                                current.updateContextRef();
                                            }
                                            catch (final ArrayIndexOutOfBoundsException e)
                                            {
                                                System.err.println(describeMCTS());
                                                throw e;
                                            }
                                            finally
                                            {
                                                prevNode.getLock().unlock();
                                            }
                                        }

                                        // If value is proven, update game theoretical values
                                        if (this.useSolver &&
                                                (current.isValueProven(
                                                current.contextRef().state().playerToAgent(
                                                        current.contextRef().state().mover())))){

                                            boolean updateGRAVE = (this.backpropFlags & 1) != 0;
                                            boolean updateGlobalActionStats = (this.backpropFlags & 2) != 0;
                                            boolean updateGlobalNGramActionStats = (this.backpropFlags & 4) != 0;
                                            List<MoveKey> moveKeysAMAF = new ArrayList();
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

                                            if (!endTrial.over() && playoutValueWeight > 0.0)
                                            {
                                                // Did not reach a terminal game state yet

                                                /********************************
                                                 Play-out
                                                 ********************************/

                                                final int numActionsBeforePlayout = current.contextRef().trial().numMoves();

                                                endTrial = playoutStrategy.runPlayout(this, playoutContext);
                                                numPlayoutActions = (endTrial.numMoves() - numActionsBeforePlayout);

                                                lastNumPlayoutActions +=
                                                        (playoutContext.trial().numMoves() - numActionsBeforePlayout);
                                            }
                                            else
                                            {
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
                                }
                                catch (final Exception e)
                                {
                                    System.err.println("MCTS error in game: " + context.game().name());
                                    e.printStackTrace();	// Need to do this here since we don't retrieve runnable's Future result
                                }
                                finally
                                {
                                    numThreadsBusy.decrementAndGet();
                                    latch.countDown();
                                }
                            }
                    );
        }

        try
        {
            latch.await(stopTime - startTime + 2000L, TimeUnit.MILLISECONDS);
        }
        catch (final InterruptedException e)
        {
            e.printStackTrace();
        }

        lastNumMctsIterations = numIterations.get();

        final Move returnMove = finalMoveSelectionStrategy.selectMove(this, rootThisCall);
        int playedChildIdx = -1;

        if (!wantsInterrupt)
        {
            int moveVisits = -1;

            for (int i = 0; i < rootThisCall.numLegalMoves(); ++i)
            {
                final BaseNode child = rootThisCall.childForNthLegalMove(i);

                if (child != null)
                {
                    if (rootThisCall.nthLegalMove(i).equals(returnMove))
                    {
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
        }
        else
        {
            analysisReport = null;
        }

        // We can already try to clean up a bit of memory here
        // NOTE: from this point on we have to use rootNode instead of rootThisCall again!
        if (!preserveRootNode)
        {
            if (!treeReuse)
            {
                rootNode = null;	// clean up entire search tree
            }
            else if (!wantsInterrupt)	// only clean up if we didn't pause the AI / interrupt it
            {
                // Keep track of entire game if node is not declared
                if (initialGameStateRoot == null){
                    initialGameStateRoot = rootNode;
                }

                // Before deleting the node, perform an analysis
                new MCTSAnalyzer(rootNode, explorationConstant, influenceEstimatedValues);

                if (playedChildIdx >= 0)
                    rootNode = rootThisCall.childForNthLegalMove(playedChildIdx);
                else
                    rootNode = null;

                if (rootNode != null)
                {
                    rootNode.setParent(null);
                    ++lastActionHistorySize;
                }
            }
        }

        return returnMove;
    }
}

/**
 * Performs an analysis on the given root node used for a search
 */
class MCTSAnalyzer{

    //-------------------------------------------------------------------------

    /** Maximum depth found */
    int maxDepth;

    /** Collected data per depth */
    ArrayList<MCTSNodeData>[] data;

    /** Exploration constant used */
    protected double explorationConstant;

    /** Influence on estimated values used */
    protected double influenceEstimatedValues;

    //-------------------------------------------------------------------------

    public MCTSAnalyzer(BaseNode rootNode, double explorationConstant, double influenceEstimatedValues){
        // Initialise parameters
        this.explorationConstant = explorationConstant;
        this.influenceEstimatedValues = influenceEstimatedValues;

        // Initialise data list for every level
        this.maxDepth = this.extractMaximumDepth(rootNode);
        this.data = new ArrayList[this.maxDepth+1]; //+1 since starts with depth 0
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = new ArrayList<>();
        }

        // Extract data from root node
        this.createData(rootNode, 0);

        // Print analysis
        this.analyse(rootNode instanceof implicitNode);
    }

    /**
     * Print all statistics of search performed from the current root node
     *
     * @param implicit true of implicit uct has been used
     */
    public void analyse(boolean implicit){
        System.out.println("#############################");
        this.countNodes();

        this.getMeasuresPerDepth(MCTSNodeData.extractData.VISITS);
        this.getMeasuresPerDepth(MCTSNodeData.extractData.TOTALSCORES);
        this.getMeasuresPerDepth(MCTSNodeData.extractData.WINRATE);
        if (implicit){
            this.getMeasuresPerDepth(MCTSNodeData.extractData.BESTSCORE);
            this.getMeasuresPerDepth(MCTSNodeData.extractData.INITIALSCORE);
        }
        this.getMeasuresPerDepth(MCTSNodeData.extractData.UCT);
    }

    /**
     * Counts the nodes, both total and active (a node is active when it has at least one child)s
     */
    private void countNodes(){
        // Create empty arrays
        int[] allNodes = new int[this.maxDepth+1];
        int[] allNonLeafNodes = new int[this.maxDepth+1];

        // Set the first to one
        allNodes[0] = 1;

        // Count the number of nodes
        for (int d = 0; d < this.maxDepth; d++){
            int count = 0;
            for (int n = 0; n < this.data[d].size(); n++) {
                count += this.data[d].get(n).numVisits.length;
            }

            allNodes[d+1] = count;
            allNonLeafNodes[d] = this.data[d].size();
        }

        // Print the number of nodes
        System.out.println("Total nodes: " + Arrays.toString(allNodes) + "\n" +
                "Active nodes: " + Arrays.toString(allNonLeafNodes) + "\n");
    }

    /**
     * Recursive function to extract data from the node and add it to the stored data
     *
     * @param node Node to extract data from
     * @param depth Search depth (starting at 0 in the root node)
     */
    private void createData(BaseNode node, int depth){
        if (hasActiveChildren(node)){
            this.data[depth].add(new MCTSNodeData(node, depth, explorationConstant, influenceEstimatedValues));

            int numChildren = node.numLegalMoves();
            for (int i = 0; i < numChildren; i++) {
                BaseNode child = node.childForNthLegalMove(i);
                if (child != null){
                    this.createData(child, depth+1);
                }
            }
        }
    }

    /**
     * Calculate and print statistics (mean, variance and sum) from the given measure for all depths
     *
     * @param selection Selection of the measure to use
     */
    private void getMeasuresPerDepth(MCTSNodeData.extractData selection){
        double[] meanSum = new double[this.maxDepth];
        double[] varianceSum = new double[this.maxDepth];
        double[] medianSum = new double[this.maxDepth];
        int[] count = new int[this.maxDepth];
        for (int d = 0; d < this.maxDepth; d++) {
            int numNodes = this.data[d].size();
            count[d] += numNodes;
            for (int n = 0; n < numNodes; n++) {
                double[] usedData = this.data[d].get(n).getData(selection);
                double mean = this.getMean(usedData);
                meanSum[d] += mean;
                varianceSum[d] += this.getVariance(usedData, mean);
                medianSum[d] += this.getMedian(usedData);
            }
        }

        double[] meanArr = new double[this.maxDepth];
        double[] varianceArr = new double[this.maxDepth];
        double[] medianArr = new double[this.maxDepth];
        for (int d = 0; d < this.maxDepth; d++) {
            meanArr[d] = meanSum[d] / (double)count[d];
            varianceArr[d] = varianceSum[d] / (double)count[d];
            medianArr[d] = medianSum[d] / (double)count[d];

        }

        DecimalFormat df = new DecimalFormat("0.0000");

        System.out.printf("%s:\n", selection.name());
        System.out.printf("Average mean: [");
        Arrays.stream(meanArr).forEach(e -> System.out.print(df.format(e) + ", " ));
        System.out.println("]");
        System.out.printf("Average variance: [");
        Arrays.stream(varianceArr).forEach(e -> System.out.print(df.format(e) + ", " ));
        System.out.println("]");
        System.out.printf("Average median: [");
        Arrays.stream(medianArr).forEach(e -> System.out.print(df.format(e) + ", " ));
        System.out.println("]");
        System.out.println("----------------------------\n");
    }

    /**
     * Determine maximum depth from the root node
     *
     * @param node Root node of current search
     * @return The depth to which nodes have been added
     */
    private int extractMaximumDepth(BaseNode node){
        return this.extractMaximumDepth(node, 0, -1);
    }

    /**
     * Recursive function to determine the maximum depth of a given node
     *
     * @param node Node to determine depth for
     * @param depth Current depth
     * @param max_depth Maximum depth found
     * @return Maximum depth at the given node
     */
    private int extractMaximumDepth(BaseNode node, int depth, int max_depth){
        // Check if depth is better
        if (depth > max_depth){
            max_depth = depth;
        }

        // For all children, determine depth
        BaseNode child;
        for (int i = 0; i < node.numLegalMoves(); i++) {
            child = node.childForNthLegalMove(i);
            if (child != null){
                int found_depth = extractMaximumDepth(child, depth + 1, max_depth);

                if (found_depth > max_depth){
                    max_depth = found_depth;
                }
            }
        }

        return max_depth;
    }

    /**
     * Calculates the mean of a given array
     *
     * @param data Array with values
     * @return Mean of data
     */
    double getMean(double[] data){
        // The mean average
        double mean = 0.0;
        for (int i = 0; i < data.length; i++) {
            mean += data[i];
        }

        mean /= data.length;

        return mean;
    }

    /**
     * Calculates the mean of a given array
     *
     * @param data Array with values
     * @param mean Mean value of data
     * @return variance of data
     */
    double getVariance(double[] data, double mean){
        // The variance
        double variance = 0;
        for (int i = 0; i < data.length; i++) {
            variance += Math.pow(data[i] - mean, 2);
        }
        variance /= data.length;

        return variance;
    }

    /**
     * Calculates the median of a given array
     *
     * @param data Array with values
     * @return Median of data
     */
    double getMedian(double[] data){
        // Copy array
        double[] sorted_data = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            sorted_data[i] = data[i];
        }

        // Sort data and determine median
        Arrays.sort(sorted_data);
        double median;
        if (data.length % 2 == 0)
            median = ((double)sorted_data[sorted_data.length/2] + (double)sorted_data[sorted_data.length/2 - 1])/2;
        else
            median = (double) sorted_data[sorted_data.length/2];

        return median;
    }

    /**
     * Check if a node has expanded children
     *
     * @param node Node to check
     * @return True if node has expanded children
     */
    private boolean hasActiveChildren(BaseNode node){
        int numChildren = node.numLegalMoves();
        for (int i = 0; i < numChildren; i++) {
            BaseNode child = node.childForNthLegalMove(i);
            if (child != null){
                return true;
            }
        }

        return false;
    }
}

/**
 * Extracts, stores and calculates all data from a given node
 */
class MCTSNodeData{

    //-------------------------------------------------------------------------

    /** Amount of children (legal moves) of the given node */
    public int totalChildren;

    /** Total scores for all active children */
    public double[] totalScores;

    /** Number of visits for active children */
    public double[] numVisits;

    /** Search depth of node */
    public int depth;

    /** Best score found with implicit minimax back-ups for all active children */
    public double[] bestScore;

    /** Initial estimated values for all children  */
    public double[] initialEstimatedValues;

    /** UCT values for all children */
    public double[] uctValues;

    //-------------------------------------------------------------------------

    /**
     * Constructor with node, depth, exploration constant and influence estimated value
     *
     * @param node Node to extract data from
     * @param depth Depth of current node
     * @param explorationConstant Exploration constant used in UCT of MCTS
     * @param influenceEstimatedValue Influence on estimated values used in UCT of MCTS
     */
    public MCTSNodeData(BaseNode node, int depth, double explorationConstant, double influenceEstimatedValue){
        // Extract the depth
        this.depth = depth;

        // Initialise the arrays
        int activeChildren = activeChildren(node);
        this.totalScores = new double[activeChildren];
        this.numVisits = new double[activeChildren];

        // Extract needed information
        this.totalChildren = node.numLegalMoves();
        int index = 0;
        int mover = node.contextRef().state().playerToAgent(node.contextRef().state().mover());

        // If the node is implicit, initialise an array for the bestScore and intitial Score, and determine implicit UCT
        if (node instanceof implicitNode){
            this.bestScore = new double[activeChildren];
            this.initialEstimatedValues = new double[this.totalChildren];

            this.uctValues = this.UCTImplicit(node, influenceEstimatedValue, explorationConstant);
        }
        // Else extract normal UCT
        else {
            this.uctValues = this.UCT(node, explorationConstant);
        }

        // For all active children extract total scores, visits best scores
        // For all children extract best initial estimated value
        for (int i = 0; i < this.totalChildren; i++) {
            BaseNode child = node.childForNthLegalMove(i);
            if (child != null){
                this.totalScores[index] = child.totalScore(mover);
                this.numVisits[index] = child.numVisits();

                if (node instanceof implicitNode){
                    this.bestScore[index] = ((implicitNode)child).getBestEstimatedValue();
                }

                index++;
            }

            if (node instanceof implicitNode){
                this.initialEstimatedValues[i] = ((implicitNode)node).getInitialEstimatedValue(i);
            }
        }
    }

    /**
     * Calculates the implicit UCT value for all children
     *
     * @param node Node to extract data from
     * @param influenceEstimatedValue Influence on the estimated values
     * @param explorationConstant Exploration constant
     * @return Implicit uct value
     */
    private double[] UCTImplicit(BaseNode node, double influenceEstimatedValue, double explorationConstant) {
        double[] values = new double[node.numLegalMoves()];

        double parentLog = Math.log((double) Math.max(1, node.sumLegalChildVisits()));
        int numChildren = node.numLegalMoves();
        State state = node.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = node.valueEstimateUnvisitedChildren(moverAgent);

        double exploit;
        double explore;
        double heuristicValue;
        for (int i = 0; i < numChildren; ++i) {
            implicitNode child = (implicitNode) node.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
                heuristicValue = ((implicitNode) node).getInitialEstimatedValue(i); // Own perspective
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double) numVisits);
                heuristicValue = moverAgent == child.contextRef().state().playerToAgent(child.contextRef().state().mover()) ?
                        child.getBestEstimatedValue() : -child.getBestEstimatedValue(); // Switch if opponent is in other perspective
            }

            values[i] = (1 - influenceEstimatedValue) * exploit +
                    influenceEstimatedValue * heuristicValue +
                    explorationConstant * explore;
        }

        return values;
    }

    /**
     * Calculates the implicit UCT value for all children
     *
     * @param node Node to extract data from
     * @param explorationConstant Exploration constant
     * @return Implicit uct value
     */
    private double[] UCT(BaseNode node, double explorationConstant) {
        double[] values = new double[node.numLegalMoves()];

        double parentLog = Math.log((double) Math.max(1, node.sumLegalChildVisits()));
        int numChildren = node.numLegalMoves();
        State state = node.contextRef().state();
        int moverAgent = state.playerToAgent(state.mover());
        double unvisitedValueEstimate = node.valueEstimateUnvisitedChildren(moverAgent);

        double exploit;
        double explore;
        for (int i = 0; i < numChildren; ++i) {
            BaseNode child = node.childForNthLegalMove(i);
            if (child == null) {
                exploit = unvisitedValueEstimate;
                explore = Math.sqrt(parentLog);
            } else {
                exploit = child.exploitationScore(moverAgent);
                int numVisits = child.numVisits() + child.numVirtualVisits();
                explore = Math.sqrt(parentLog / (double) numVisits);
            }

            values[i] = exploit + explorationConstant * explore;
        }

        return values;
    }

    /**
     * Calculates the win rate based on the total score and number of visits for all
     * active children
     * @return win rates for all active children
     */
    private double[] calculateWinRate(){
        double[] values = new double[this.totalScores.length];
        for (int i = 0; i < values.length; i++) {
            values[i] = this.totalScores[i] / this.numVisits[i];
        }

        return values;
    }

    /**
     * Getter for the selected data
     *
     * @param selection Indicates which data to get
     * @return Requested data
     */
    public double[] getData(extractData selection){
        switch (selection){
            case VISITS -> {
                return this.numVisits;
            }
            case TOTALSCORES -> {
                return this.totalScores;
            }
            case BESTSCORE -> {
                return this.bestScore;
            }
            case INITIALSCORE -> {
                return this.initialEstimatedValues;
            }
            case UCT -> {
                return this.uctValues;
            }
            case WINRATE -> {
                return this.calculateWinRate();
            }
            default -> {
                throw new RuntimeException("Data selection not implemented.");
            }
        }
    }

    /**
     * Counts the number of expanded children in the current node
     *
     * @param node Node to extract data from
     * @return Number of expanded children
     */
    private int activeChildren(BaseNode node){
        int count = 0;
        int numChildren = node.numLegalMoves();
        for (int i = 0; i < numChildren; i++) {
            BaseNode child = node.childForNthLegalMove(i);
            if (child != null){
                count++;
            }
        }

        return count;
    }

    /**
     * Enum for getter to select data
     */
    public enum extractData {
        VISITS,
        TOTALSCORES,
        BESTSCORE,
        INITIALSCORE,
        UCT,
        WINRATE;

        private extractData() {
        }
    }
}