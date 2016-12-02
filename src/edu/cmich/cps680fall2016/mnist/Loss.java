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
            // The gradient of the MSE can be written as:
            //  L'_k(z,e) = -2 * (e_k - z_k)
            // Note that the contribution to MSE from each component of the input
            // is independent of the other components, so the components of the
            // gradient are independent of each other as well.
            for (int k = 0; k < in.length; k++) {
                out[k] = -2 * (expected[k] - in[k]);
            }
        }
    };

    /**
     * Cross-entropy of the expected distribution with respect to the actual
     * distribution.
     * <p>
     * Note that this <b>requires</b> the input vectors to be strictly in the
     * range (0,1].
     * <p>
     * This loss function is theoretically very useful, but suffers from severe
     * issues with numerical instability.
     */
    public static Loss CROSS_ENTROPY = new Loss() {

        @Override public float loss(float[] in, float[] expected) {
            // The cross-entropy of e with respect to z is defined as:
            //  L(e,z) = - SUM over all k of: e_k * log2(z_k)
            double loss = 0;
            for (int k = 0; k < in.length; k++) {
                loss -= expected[k] * Math.log(in[k]);
            }
            loss /= Math.log(2);
            return (float) loss;
        }

        @Override public void gradient(float[] in, float[] expected, float[] out) {
            // The gradient of the cross-entropy of e with respect to z is:
            //  L'_k(e,z) = - (1/log(2)) e_k / z_k
            // As with MSE, the components of the gradient are independent.
            // NOTE: This function suffers from severe numerical instabilities
            final float niln2 = (float) (-1 / Math.log(2));
            for (int k = 0; k < in.length; k++) {
                if (expected[k] == 0) out[k] = 0;
                else out[k] = (niln2 * expected[k]) / in[k];
            }
        }
    };

    /**
     * Cross-entropy of the expected distribution with respect to the softmax of
     * the input.
     * <p>
     * This loss function combines softmax normalization with cross-entropy
     * loss, allowing for more numerically stable calculations.
     */
    public static Loss SOFTMAX_CROSS_ENTROPY = new Loss() {

        @Override public float loss(float[] in, float[] expected) {
            // The cross-entropy of e with respect to the softmax of z:
            //  L(e,z) = - SUM over all k of: e_k * log2(g_k(z))
            // where:
            //  g_k(z) = exp(z_k) / (SUM over all i of: exp(z_i))
            // and thus:
            //       N = SUM over all i of: exp(z_i)
            //  L(e,z) = - (1/log(2)) SUM over all k of: e_k * log(exp(z_k) / N)
            //         = - (1/log(2)) SUM over all k of: e_k * log(exp(z_k)) - log(N)
            //         = - (1/log(2)) SUM over all k of: e_k * z_k - log(N)
            // or, finally:
            //  L(e,z) = (1/log(2)) * (count(z)*log(N) - SUM over all k of: e_k * z_k)
            double norm = 0, dot = 0;
            for (int k = 0; k < in.length; k++) {
                norm += Math.exp(in[k]);
                dot += expected[k] * in[k];
            }
            double loss = (in.length * Math.log(norm) - dot) / Math.log(2);
            return (float) loss;
        }

        @Override public void gradient(float[] in, float[] expected, float[] out) {
            // The gradient of the cross-entropy of e with respect to the softmax of z:
            //  L'_k(e,z) = - (1/log(2)) * SUM over all k!=j of: (e_j * g_k(z) - e_k * g_j(z))
            // where:
            //     g_k(z) = exp(z_k) / (SUM over all i of: exp(z_i))
            double norm = 0;
            double[] ein = new double[in.length];
            for (int k = 0; k < in.length; k++) {
                norm += (ein[k] = Math.exp(in[k]));
            }
            norm *= -Math.log(2);
            for (int j = 0; j < out.length; j++) {
                final double ein_j = ein[j];
                double dg_j = 0;
                for (int k = 0; k < expected.length; k++) {
                    if (k == j) continue;
                    dg_j += expected[j] * ein[k] - expected[k] * ein_j;
                }
                out[j] = (float) (dg_j / norm);
            }
        }
    };
}
