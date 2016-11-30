package edu.cmich.cps680fall2016.mnist;

/** Activation function for NN evaluation. */
public interface Activation {

    /**
     * Compute the activation function for the input vector, and place the
     * output in the output vector. Input and output must be the same size, but
     * each component of the output may depend on any or all components of the
     * input.
     */
    public void activate(float[] in, float[] out);

    /**
     * Compute the directional derivative of the activation function, i.e. the
     * inner product of the gradient of the activation function (a tensor) with
     * a direction vector. Input, direction, and output vectors must have the
     * same size.
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
    };

    /** Logistic activation */
    public static Activation LOGISTIC = new Activation() {

        @Override public void activate(float[] in, float[] out) {
            for (int j = 0; j < out.length; j++) {
                out[j] = (float) (1 / (1 + Math.exp(-in[j])));
            }
        }

        @Override public void dctDerivative(float[] in, float[] dir, float[] out) {
            // Each component of logistic output depends only on the corresponding
            // component of the input.  Therefore all off-diagonal elements of 
            // the gradient tensor are zero, and the directional derivative is
            // just the regular derivative dotted with the direction vector.
            for (int j = 0; j < out.length; j++) {
                double val = 1 / (1 + Math.exp(-in[j]));
                out[j] = (float) (val * (1 - val) * dir[j]);
            }
        }
    };

    /** Softmax activation, normalized exponential outputs */
    public static Activation SOFTMAX = new Activation() {

        @Override public void activate(float[] in, float[] out) {
            double norm = 0;
            for (int j = 0; j < out.length; j++) {
                double v = Math.exp(in[j]);
                out[j] = (float) v;
                norm += v;
            }
            for (int j = 0; j < out.length; j++) {
                out[j] /= norm;
            }
        }

        @Override public void dctDerivative(float[] in, float[] dir, float[] out) {
            // Each component of softmax output depends on all components on input.
            // Therefore gradient is a full 2D tensor and directional derivative
            // requires a matrix * vector multiplication.
            double norm = 0;
            for (int k = 0; k < in.length; k++) {
                double v = Math.exp(in[k]);
                out[k] = (float) v;
                norm += v;
            }
            for (int j = 0; j < out.length; j++) {
                double out_j = 0;
                for (int k = 0; k < in.length; k++) {
                    final double g_k = Math.exp(in[k]) / norm;
                    double dg_kj = (j == k) //
                    ? dg_kj = g_k * (1 - g_k)
                            : -g_k * (Math.exp(in[j]) / norm);
                    out_j += dir[k] * dg_kj;
                }
                out[j] = (float) out_j;
            }
        }
    };
}
