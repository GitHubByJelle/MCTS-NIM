package World;

import Agents.*;
import app.StartDesktopApp;
import manager.ai.AIRegistry;

/**
 * Main class which can be used to load the GUI of Ludii, while adding own bots
 */
public class LaunchLudii {
    /**
     * Main class which can be used to load the GUI of Ludii, while adding own bots
     *
     * @param args No input required
     */
    public static void main(final String args[]) {
        AIRegistry.registerAI
                (
                        "Example random AI",
                        () -> {
                            return new RandomAI();
                        },
                        (game) -> {
                            return true;
                        }
                );

        AIRegistry.registerAI
                (
                        "UBFM - Cohen-Solal",
                        () -> {
                            return new UBFMHF();
                        },
                        (game) -> {
                            return new UBFMNN().supportsGame(game);
                        }
                );

        AIRegistry.registerAI
                (
                        "Descent Heuristic Function",
                        () -> {
                            return new descentHF();
                        },
                        (game) -> {
                            return new descentHF().supportsGame(game);
                        }
                );

        AIRegistry.registerAI
                (
                        "Completed descent",
                        () -> {
                            return new descentNN();
                        },
                        (game) -> {
                            return new descentNN().supportsGame(game);
                        }
                );

        AIRegistry.registerAI
                (
                        "AlphaBeta with HF",
                        () -> {
                            return new AlphaBetaSearchHF();
                        },
                        (game) -> {
                            return new AlphaBetaSearchHF().supportsGame(game);
                        }
                );

        AIRegistry.registerAI
                (
                        "ExampleUCT",
                        () -> {
                            return new ExampleUCT();
                        },
                        (game) -> {
                            return new ExampleUCT().supportsGame(game);
                        }
                );

        AIRegistry.registerAI
                (
                        "UBFM Completed (Heuristic)",
                        () -> {
                            return new UBFMHFCompleted();
                        },
                        (game) -> {
                            return new UBFMHFCompleted().supportsGame(game);
                        }
                );

        AIRegistry.registerAI
                (
                        "UBFM Completed (NN)",
                        () -> {
                            return new UBFMNNCompleted("NN_models/Network_bSize128_nEp1_nGa4034_2022-11-10-07-58-23.bin");
                        },
                        (game) -> {
                            return new UBFMNNCompleted().supportsGame(game);
                        }
                );

        AIRegistry.registerAI
                (
                        "MCTSSolver",
                        () -> {
                            return new MCTSSolver();
                        },
                        (game) -> {
                            return true;
                        }
                );

        AIRegistry.registerAI
                (
                        "Implicit MCTS",
                        () -> {
                            return new ImplicitMCTS();
                        },
                        (game) -> {
                            return true;
                        }
                );

        AIRegistry.registerAI
                (
                        "Implicit MCTS Solver",
                        () -> {
                            return new ImplicitMCTSSolver();
                        },
                        (game) -> {
                            return true;
                        }
                );

        // Launch Ludii
        StartDesktopApp.main(new String[0]);
    }
}
