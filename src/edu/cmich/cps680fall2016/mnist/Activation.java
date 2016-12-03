package edu.cmich.cps680fall2016.mnist;

/** Activation function for NN evaluation. */
public interface Activation {

    /**
     * Compute the activation function for the input vector, and place the
     * output in the output vector. Input and output must be the same size, but
     * each component of the output may depend on any or all components of the
     * input.
     * 
     * <b>Note:</b> {@code in} and {@code out} may refer to the same array
     * object.
     */
    public void activate(float[] in, float[] out);

    /**
     * Compute the directional derivative of the activation function, i.e. the
     * inner product of the gradient of the activation function (a tensor) with
     * a direction vector. Input, direction, and output vectors must have the
     * same size.
     * 
     * <b>Note:</b> {@code in} and {@code out} may refer to the same array
     * object.
     */
    public default void dctDerivative(float[] in, float[] dir, float[] out) {
        throw new UnsupportedOperationException("Not implemened yet.");
    }

    /** Passthrough activation, i.e. a no-op. */
    public static Activation PASSTHROUGH = new Activation() {

        @Override public void activate(float[] in, float[] out) {
            if (in == out) return;
            System.arraycopy(in, 0, out, 0, in.length);
        }

        @Override public void dctDerivative(float[] in, float[] dir, float[] out) {
            if (dir == out) return;
            System.arraycopy(dir, 0, out, 0, dir.length);
        }

        @Override public String toString() {
            return "passthrough";
        }
    };

    /** Logistic activation */
    public static Activation LOGISTIC = new Activation() {

        @Override public void activate(float[] in, float[] out) {
            // The scalar logistic function is g(z) = 1/(1 + exp(z)).
            // The vector logistic function is just the scalar logistic function
            // applied to each component of the vector independently:
            //   g_k(z) = 1/(1 + exp(z_k))
            for (int k = 0; k < out.length; k++) {
                out[k] = (float) (1 / (1 + Math.exp(-in[k])));
            }
        }

        @Override public void dctDerivative(float[] in, float[] dir, float[] out) {
            // The derivative of the scalar logistic function g(z) can be written
            // as g'(z) = g(z) * (1 - g(z)).
            // Therefore the gradient tensor of the vector logistic function can
            // be written as:
            //  g'_jk(z) = g_k(z) * (1 - g_k(z))    if j==k 
            //           = 0                        if j!=k
            // Note that the off-diagonal components of the tensor are zero,
            // this is true for all vector functions that treat each component
            // of the vector independently.
            // The directional derivative is therefore just the diagonal components
            // dotted with the direction vector:
            //  dg_j(z) = d_j * g_j(z) * (1 - g_j(z))
            for (int j = 0; j < out.length; j++) {
                double val = 1 / (1 + Math.exp(-in[j]));
                out[j] = (float) (val * (1 - val) * dir[j]);
            }
        }
        
        @Override public String toString() {
            return "logistic";
        }
    };

    /** Softmax activation */
    public static Activation SOFTMAX = new Activation() {

        @Override public void activate(float[] in, float[] out) {
            // The softmax function exponentiates each component of the input
            // and the normalizes the result:
            //  g_k(z) = exp(z_k) / (SUM over all i of: exp(z_i))
            double norm = 0;
            for (int k = 0; k < out.length; k++) {
                norm += Math.exp(in[k]);
            }
            for (int k = 0; k < out.length; k++) {
                out[k] = (float) (Math.exp(in[k]) / norm);
            }
        }

        @Override public void dctDerivative(float[] in, float[] dir, float[] out) {
            // The gradient tensor of the softmax function can be written as:
            //  g'_jk(z) = g_k(z) * (1 - g_k(z))    if j==k 
            //           = - g_k(z) * g_j(z)        if j!=k 
            // The directional derivative is the left-product of the direction
            // vector with this tensor:
            //  dg_j(z) = SUM over all k of: d_k * g'_jk(z)
            //          = d_j * g'_jj(z) + SUM over all k!=j of: d_k * g'_jk(z)
            //          = d_j * g_j(z) * (1 - g_j(z))
            //              - SUM over all k!=j of: d_k * g_k(z) * g_j(z)
            // Note that the product  X*(1-X) is numerically unstable when
            // X is close to 1 or 0; this means that a naive implementation of
            // the above equation will tend to have stability problems.
            // After some thought, I recognized that:
            //  1 - g_j(z) = SUM over all k!=j of: g_k(z)
            // Therefore we can write:
            //  dg_j(z) = SUM over all k!=j of: d_j * g_j(z) * g_k(z)
            //              - SUM over all k!=j of: d_k * g_k(z) * g_j(z)
            // or, finally:
            //  dg_j(z) = g_j(z) * SUM over all k!=j of: g_k(z) * (d_j - d_k)
            // This is as stable as I could make it.
            double norm = 0;
            double[] ein = new double[in.length];
            for (int k = 0; k < in.length; k++) {
                norm += (ein[k] = Math.exp(in[k]));
            }
            for (int j = 0; j < out.length; j++) {
                final double g_j = ein[j] / norm;
                double dg_j = 0;
                for (int k = 0; k < dir.length; k++) {
                    if (k == j) continue;
                    final double g_k = ein[k] / norm;
                    dg_j += g_k * (dir[j] - dir[k]);
                }
                out[j] = (float) (g_j * dg_j);
            }
        }
        
        @Override public String toString() {
            return "softmax";
        }
    };
}
