package MCTSStrategies.Rescaler;

import utils.Value;

/**
 * Class which rescales the given values between a fixed minimum and maximum bound
 */
public class FixedBounds implements Rescaler {

    //-------------------------------------------------------------------------

    /**
     * Maximum bound of rescaled values
     */
    float maxBound;

    /**
     * Minimum bound of rescaled values
     */
    float minBound;

    //-------------------------------------------------------------------------

    /**
     * Constructor for the fixed bounds rescaler
     *
     * @param maxBound Maximum bound of rescaled values
     * @param minBound Minimum bound of rescaled values
     */
    public FixedBounds(float maxBound, float minBound) {
        this.maxBound = maxBound;
        this.minBound = minBound;
    }

    /**
     * Rescales the given values between the given bounds while keeping the relative difference between
     * the values similar
     *
     * @param values values to rescale
     * @return rescaled values with fixed bounds
     */
    @Override
    public double[] rescale(double[] values) {
        // Initialise needed variables
        double min = Value.INF;
        double max = -Value.INF;
        int n = values.length;

        // Determine mix and max value
        for (int i = 0; i < n; i++) {
            double v = values[i];
            if (v > max) {
                max = v;
            } else if (v < min) {
                min = v;
            }
        }

        // Determine multiplier
        double multiplier = (maxBound - minBound) / (max - min);

        // Determine new values
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = (values[i] - min) * multiplier + minBound;
        }

        return result;
    }
}
