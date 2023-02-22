package MCTSStrategies.Rescaler;

/**
 * Class which rescales the given values by multiplying the differences to the mean with the given multiplier
 */
public class MultiplyDifferences implements Rescaler {

    //-------------------------------------------------------------------------

    /**
     * Multiplier that is used to multiply the differences (w.r.t. the mean) with
     */
    float multiplier;

    //-------------------------------------------------------------------------

    /**
     * Constructor for the multiplied differences
     *
     * @param multiplier Multiplier that is used to multiply the differences (w.r.t. the mean) with
     */
    public MultiplyDifferences(float multiplier) {
        this.multiplier = multiplier;
    }

    /**
     * Rescales the given values by multiplying the differences w.r.t. the mean
     *
     * @param values values to rescale
     * @return rescaled values with multiplied difference
     */
    @Override
    public double[] rescale(double[] values) {
        // Calculate mean
        double sum = 0;
        int n = values.length;
        for (int i = 0; i < n; i++) {
            sum += values[i];
        }

        double mean = sum / n;

        // Determine new values with multiplied difference
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = (values[i] - mean) * this.multiplier + mean;
        }

        return result;
    }
}
