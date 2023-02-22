package MCTSStrategies.Rescaler;

/**
 * Class which rescales the given values to softmax
 */
public class Softmax implements Rescaler {

    //-------------------------------------------------------------------------

    /**
     * Lower bound for the Temperature (if to small value is used, Math.exp will return inf values)
     */
    double lowerBoundT = .05;

    //-------------------------------------------------------------------------

    /**
     * Constructor for the softmax rescaler
     */
    public Softmax() {
    }

    /**
     * Rescales the given values using the softmax with Temperature of 1
     *
     * @param values values to rescale
     * @return rescaled values using softmax
     */
    @Override
    public double[] rescale(double[] values) {
        return rescale(values, 1);
    }

    /**
     * Rescales the given values using the softmax with Temperature of 1
     *
     * @param values values to rescale
     * @param T      Temperature of softmax
     * @return rescaled values using softmax
     */
    public double[] rescale(double[] values, double T) {
        // Prevent the temperature from going to large since it can result in problems w.r.t. Math.exp
        T = Math.max(T, lowerBoundT);

        // Calculate sum
        double sum = 0;
        int n = values.length;
        for (int i = 0; i < n; i++) {
            sum += (float) Math.exp(values[i] / T);
        }

        // Determine probabilities
        double[] probabilities = new double[n];
        for (int i = 0; i < n; i++) {
            probabilities[i] = Math.exp(values[i] / T) / sum;
        }

        return probabilities;
    }
}
