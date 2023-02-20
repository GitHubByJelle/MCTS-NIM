package MCTSStrategies.Rescaler;

/**
 * Class to "rescale" the given values linear (used as default class)
 * Can be used when a Rescaler class is required, but no rescaling is required
 */
public class Linear implements Rescaler{
    /**
     * Constructor
     */
    public Linear(){}

    /**
     * Returns the same values as the given values
     *
     * @param values values
     * @return same values
     */
    @Override
    public double[] rescale(double[] values) {
        return values;
    }
}
