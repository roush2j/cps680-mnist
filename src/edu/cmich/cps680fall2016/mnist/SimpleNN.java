package edu.cmich.cps680fall2016.mnist;

import java.io.FileOutputStream;
import java.util.Random;

public class SimpleNN {

    /** The number of inputs/neurons in each layer */
    private final int[] shape;

    /**
     * Weight matrices for each successive pair of layers (0->1, 1->2, ...).
     * Each element {@code i} has size {@code (shape[i] + 1) * shape[i+1]},
     * containing weights in {@code [i]-major} order.
     */
    private final float[][] weights;

    /** The activation function between each successive pair of layers. */
    private final Activation[] actv;

    /**
     * Create a new NN with all weights initialized to 0.
     * 
     * @param shape An array containing the size of each layer in the NN.
     * @param actvFuncs An array containing the activation function between each
     *            layer.
     */
    public SimpleNN(int[] shape, Activation[] actvFuncs) {
        assert (shape.length >= 2);
        assert (actvFuncs.length == shape.length - 1);

        this.shape = shape.clone();
        this.weights = new float[shape.length - 1][];
        for (int layeridx = 0; layeridx < shape.length - 1; layeridx++) {
            int wcnt = (shape[layeridx] + 1) * shape[layeridx + 1];
            weights[layeridx] = new float[wcnt];
        }
        this.actv = actvFuncs.clone();
    }

    /**
     * Create a new NN with all weights initialized using a Gaussian
     * distribution with standard deviation of 0.5.
     * 
     * @see #SimpleNN(int[], Activation[])
     */
    public SimpleNN(int[] shape, Activation[] actvFuncs, Random rand) {
        this(shape, actvFuncs);

        // initialize weights with random values
        for (float[] w : weights) {
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) rand.nextGaussian() / 2;
            }
        }
    }

    /**
     * Allocate and return value storage arrays for training/applying the NN.
     * 
     * @return A 2D array suitable for passing to {@link #apply(float[][])} or
     *         {@link #train(float[][])}.
     */
    public float[][] valueArray() {
        float[][] values = new float[shape.length][];
        for (int layeridx = 0; layeridx < shape.length; layeridx++) {
            values[layeridx] = new float[shape[layeridx]];
        }
        return values;
    }

    /**
     * Apply the neural network to an input.
     * 
     * @param values An array of values from each layer of the network, where
     *            {@code values[0]} is the input to the first layer and
     *            {@code values[values.length - 1]} is the output from the last
     *            layer.
     */
    public void apply(float[][] values) {
        assert (values.length == shape.length);
        assert (values[0].length == shape[0]);

        /*
         * Reference Example w/ 3 'layers':
         *      l0 = 3:                        l1 = 2:            l2 = 1:
         *                  w00 w01
         *                  w02 w03                    w10
         *                  w04 w05                    w11
         *                x w06 w07                  x w12 
         *  (1) v00 v01 v02 --------ACTV-> (1) v10 v11 ----ACTV-> v20
         */

        // apply NN
        for (int layeridx = 0; layeridx < weights.length; layeridx++) {
            assert (values[layeridx + 1].length == shape[layeridx + 1]);

            // compute weighted sum for next layer
            final float[] w = weights[layeridx];
            final float[] v = values[layeridx];
            final float[] nv = values[layeridx + 1];
            final int shapel = shape[layeridx], shapeln = shape[layeridx + 1];
            System.arraycopy(w, 0, nv, 0, shapeln); // bias
            for (int biased_ij = shapeln, i = 0; i < shapel; i++) {
                final float v_i = v[i];
                for (int j = 0; j < shapeln; j++, biased_ij++) {
                    nv[j] += w[biased_ij] * v_i;
                }
            }

            // compute activation function for next layer
            actv[layeridx].activate(nv, nv);
        }
    }

    /**
     * Train the neural network with an input, output pair.
     * 
     * @param values An array of values from each layer of the network, where
     *            {@code values[0]} is the input to the first layer and
     *            {@code values[values.length - 1]} is the <b>expected</b>
     *            output from the last layer.
     */
    public void train(float[][] values) {
        // ... TODO
    }

    public void dump() {
        for (int w = 0; w < weights.length; w++) {
            for (int q = 0; q < shape[w + 1]; q++) {
                int R = 28, C = 28;
                DispImage img = new DispImage(DispImage.floatPix(weights[w])
                        .stride(q, shape[w + 1]).normalize(-1, 1), R, C);
                String name = String.format("weight-%02d-%02d.png", w, q);
                try {
                    img.writePNG(new FileOutputStream(name));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /** Activation function for NN evaluation. */
    public static interface Activation {

        /**
         * For each value in {@code values}, write an activation output in
         * {@code out} in the range [0,1].
         */
        public void activate(float[] values, float[] out);
    }

    /** Passthrough activation, i.e. a no-op. */
    public static Activation PASSTHROUGH = new Activation() {

        @Override public void activate(float[] values, float[] out) {
            if (values == out) return;
            System.arraycopy(values, 0, out, 0, values.length);
        }
    };

    /** Threshold activation, i.e. the Heaviside step function. */
    public static Activation THRESHOLD = new Activation() {

        @Override public void activate(float[] values, float[] out) {
            for (int i = 0; i < values.length; i++) {
                out[i] = values[i] > 0 ? 1 : 0;
            }
        }
    };

    /** Logistic activation */
    public static Activation LOGISTIC = new Activation() {

        @Override public void activate(float[] values, float[] out) {
            for (int i = 0; i < values.length; i++) {
                out[i] = (float) (1 / (1 + Math.exp(-values[i])));
            }
        }
    };

    /** Softmax activation, normalized exponential outputs */
    public static Activation SOFTMAX = new Activation() {

        @Override public void activate(float[] values, float[] out) {
            double norm = 0;
            for (int i = 0; i < values.length; i++) {
                double v = Math.exp(values[i]);
                out[i] = (float) v;
                norm += v;
            }
            for (int i = 0; i < values.length; i++) {
                out[i] /= norm;
            }
        }
    };

}
