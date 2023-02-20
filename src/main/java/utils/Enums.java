package utils;

/**
 * All Enums used during implementations.
 */
public class Enums {
    /**
     * Enum for the exploration policy (selection during search) of UBFM as indicated by Cohen in:
     * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
     * knowledge. arXiv preprint arXiv:2008.01188.
     */
    public static enum ExplorationPolicy {
        BEST,
        EPSILON_GREEDY,
        SOFTMAX;

        private ExplorationPolicy() {
        }
    }

    /**
     * Enum for the final move selection policy of UBFM as indicated by Cohen in:
     * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
     * knowledge. arXiv preprint arXiv:2008.01188.
     */
    public static enum SelectionPolicy {
        BEST,
        SAFEST;

        private SelectionPolicy() {
        }
    }

    /**
     * Enum for the data selection strategy used from the descent framework as proposed in:
     * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
     * knowledge. arXiv preprint arXiv:2008.01188.
     */
    public static enum DataSelection {
        TREE,
        ROOT,
        TERMINAL;

        private DataSelection() {
        }
    }

    /**
     * Enum for the terminal evaluation function used of UBFM as proposed by Cohen in:
     * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
     * knowledge. arXiv preprint arXiv:2008.01188.
     */
    public static enum TerminalEvaluationSelection {
        CLASSIC,
        DEPTH;

        private TerminalEvaluationSelection() {
        }
    }

    /**
     * Enum for the Neural Network architecture used.
     * "Cohen" is the architecture proposed in:
     * Cohen-Solal, Q. (2020). Learning to play two-player perfect-information games without
     * knowledge. arXiv preprint arXiv:2008.01188.
     * "TicTacToe" is contains less CNN layers to fit the game better.
     */
    public enum NetworkType {
        TicTacToe,
        Cohen;

        private NetworkType() {
        }
    }
}
