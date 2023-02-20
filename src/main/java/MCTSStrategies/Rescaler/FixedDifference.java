package MCTSStrategies.Rescaler;

import utils.Value;

/**
 * Class which rescales the given values between minimum and maximum with a fixed difference located around the
 * mean of the given values
 */
public class FixedDifference implements Rescaler{

    //-------------------------------------------------------------------------

    /** Allowed fixed difference between minimum and maximum value */
    float fixedDifference;

    //-------------------------------------------------------------------------

    /**
     * Constructor for the fixed difference rescaler
     *
     * @param fixedDifference Allowed fixed difference between minimum and maximum value
     */
    public FixedDifference(float fixedDifference){
        this.fixedDifference = fixedDifference;
    }

    /**
     * Rescales the given values between the minimum and maximum value with a fixed bound
     *
     * @param values values to rescale
     * @return rescaled values with fixed difference
     */
    @Override
    public double[] rescale(double[] values) {
        // Initialise needed variables
        double sum = 0;
        double min = Value.INF;
        double max = -Value.INF;
        int n = values.length;

        // Determine min, max and sum
        for (int i = 0; i < n; i++) {
            double v = values[i];
            sum += v;

            if (v > max){
                max = v;
            } else if (v < min) {
                min = v;
            }
        }

        // Determine mean and multiplier
        double mean = sum / n;
        double multiplier = this.fixedDifference / (max - min);

        // Determine new vales
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = (values[i] - mean) * multiplier + mean;
        }

        return result;
    }
}
