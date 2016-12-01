package edu.cmich.cps680fall2016.mnist;

/** Loss function for NN evaluation. */
public interface Loss {

    /**
     * Compute the loss of the input with respect to the expected values. Input
     * and expected values must be same size.
     */
    public float loss(float[] in, float[] expected);

    /**
     * Compute the gradient of the loss of the input with respect to the
     * expected values. Input, expected values, and output must be the same
     * size.
     * 
     * <b>Note:</b> {@code in} and {@code out} may refer to the same array
     * object.
     */
    public void gradient(float[] in, float[] expected, float[] out);

    /** Average of squared errors */
    public static Loss MEAN_SQUARED_ERR = new Loss() {

        @Override public float loss(float[] in, float[] expected) {
            double loss = 0;
            for (int k = 0; k < in.length; k++) {
                double err = expected[k] - in[k];
                loss += err * err;
            }
            return (float) loss;
        }

        @Override public void gradient(float[] in, float[] expected, float[] out) {
            // Note that the contribution to MSE from each component of the input
            // is independent of the other components, so the components of the
            // gradient are independent of each other as well.
            for (int k = 0; k < in.length; k++) {
                out[k] = -2 * (expected[k] - in[k]);
            }
        }
    };

    /**
     * Cross-entropy of the expected and actual distributions.
     * <p>
     * Note that this <b>requires</b> the input and expected vectors to be be
     * strictly in the range (0,1].
     * <p>
     * This loss function is theoretically very useful, but suffers from severe
     * issues with numerical instability.
     */
    public static Loss CROSS_ENTROPY = new Loss() {

        @Override public float loss(float[] in, float[] expected) {
            double loss = 0;
            for (int k = 0; k < in.length; k++) {
                loss -= expected[k] * Math.log(in[k]);
            }
            loss /= Math.log(2);
            return (float) loss;
        }

        @Override public void gradient(float[] in, float[] expected, float[] out) {
            // As with MSE, the components of the gradient are independent.
            // NOTE: This function suffers from severe numerical instabilities
            final float niln2 = (float) (-1 / Math.log(2));
            for (int k = 0; k < in.length; k++) {
                if (expected[k] == 0) out[k] = 0;
                else out[k] = (niln2 * expected[k]) / in[k];
            }
        }
    };
}
