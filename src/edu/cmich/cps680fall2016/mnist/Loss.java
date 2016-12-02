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
            //  L'_j(z,e) = -2 * (e_j - z_j)
            // Note that the contribution to MSE from each component of the input
            // is independent of the other components, so the components of the
            // gradient are independent of each other as well.
            for (int j = 0; j < in.length; j++) {
                out[j] = -2 * (expected[j] - in[j]);
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
            //  L(e,z) = - SUM over all k of: e_k * log(z_k)
            double loss = 0;
            for (int k = 0; k < in.length; k++) {
                loss -= expected[k] * Math.log(in[k]);
            }
            return (float) loss;
        }

        @Override public void gradient(float[] in, float[] expected, float[] out) {
            // The gradient of the cross-entropy of e with respect to z is:
            //  L'_j(e,z) = - e_j / z_j
            // As with MSE, the components of the gradient are independent.
            // NOTE: This function suffers from severe numerical instabilities
            for (int j = 0; j < in.length; j++) {
                if (expected[j] == 0) out[j] = 0;
                else out[j] = -expected[j] / in[j];
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
            //  L(e,z) = - SUM over all k of: e_k * log(exp(z_k) / N)
            // where:
            //       N = SUM over all i of: exp(z_i)
            // and thus:
            //  L(e,z) = - SUM over all k of: e_k * (log(exp(z_k)) - log(N))
            //         = - SUM over all k of: (e_k * z_k - e_k * log(N))
            // or, finally:
            //  L(e,z) = - SUM over all k of: e_k * z_k
            //              + log(N) * SUM over all k of: e_k 
            double norm = 0, dot = 0, sum = 0;
            for (int k = 0; k < in.length; k++) {
                sum  += expected[k];
                norm += Math.exp(in[k]);
                dot += expected[k] * in[k];
            }
            double loss = sum * Math.log(norm) - dot;
            return (float) loss;
        }

        @Override public void gradient(float[] in, float[] expected, float[] out) {
            // The gradient of the normalization factor N is:
            //   L'_j(e,z) = d/dz_j( - SUM over all k of: e_k * z_k )
            //                 + d/dz_j( log(N) * SUM over all k of: e_k  )
            //             = -e_j + d/dz_j( log(N) ) * SUM over all k of: e_k
            //             = -e_j + (1/N)*d/dz_j(N) * SUM over all k of: e_k
            // Since:
            //  (d/dz_j) N = (d/dz_j) SUM over all i of: exp(z_i)
            //             = exp(z_j)
            // We have:
            //   L'_j(e,z) = -e_j + (exp(z_j)/N)*SUM over all k of: e_k
            double norm = 0, sum = 0;
            for (int k = 0; k < in.length; k++) {
                sum  += expected[k];
                norm += Math.exp(in[k]);
            }
            for (int j = 0; j < out.length; j++) {
                final double gz_j = Math.exp(in[j]) / norm;
                out[j] = (float) (gz_j * sum - expected[j]);
            }
        }
    };
}
