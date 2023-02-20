package Experiments;

import other.AI;
import other.context.Context;
import other.move.Move;
import utils.DebugTools;

/**
 * Main class which can be used to debug a bot on a specific game position.
 */
public class debugBotOnBoard {
    /**
     * Main method to debug a bot.
     *
     * @param args no inputs required
     */
    public static void main(String[] args) {
        // Load existing context
        Context context = DebugTools.loadExistingContext("BreakthroughBoard2.obj", "Breakthrough");

        // Create string for board
        System.out.println(DebugTools.contextToString(context));

        // Use agent to debug
        AI agent = new Agents.ImplicitMCTS();

        // Play a move
        int mover = context.state().mover();
        agent.initAI(context.game(), mover);
        final Move move = agent.selectAction
                (
                        context.game(),
                        new Context(context),
                        -1,
                        1000,
                        -1
                );
    }
}
