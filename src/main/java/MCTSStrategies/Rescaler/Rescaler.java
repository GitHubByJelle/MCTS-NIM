package MCTSStrategies.Rescaler;

/**
 * Interface which can be used to implement different types of rescalers
 */
public interface Rescaler {
    /**
     * Rescales the given values
     *
     * @param values values to rescale
     * @return rescaled values
     */
    double[] rescale(double[] values);
}
