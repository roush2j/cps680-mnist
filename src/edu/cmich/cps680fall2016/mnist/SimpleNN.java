package edu.cmich.cps680fall2016.mnist;

import java.util.Random;

public class SimpleNN {

    /** The number of inputs/neurons in each layer */
    public final int[] shape;

    /**
     * Weight matrices for each successive pair of layers (0->1, 1->2, ...).
     * Each element {@code i} has size {@code (shape[i] + 1) * shape[i+1]},
     * containing weights in {@code [i]-major} order.
     */
    public final float[][] weights;

    /** The activation function between each successive pair of layers. */
    public final Activation[] actv;

    /** The loss function used to evaluate the output of the final layer. */
    public final Loss loss;

    /**
     * Create a new NN with all weights initialized to 0.
     * 
     * @param shape An array containing the size of each layer in the NN.
     * @param actvFuncs An array containing the activation function between each
     *            layer.
     */
    public SimpleNN(int[] shape, Activation[] actvFuncs, Loss lossFunc) {
        assert (shape.length >= 2);
        assert (actvFuncs.length == shape.length - 1);

        this.shape = shape.clone();
        this.weights = new float[shape.length - 1][];
        for (int layeridx = 0; layeridx < shape.length - 1; layeridx++) {
            int wcnt = (shape[layeridx] + 1) * shape[layeridx + 1];
            weights[layeridx] = new float[wcnt];
        }
        this.actv = actvFuncs.clone();
        this.loss = lossFunc;
    }

    /**
     * Create a new NN with all weights initialized using a Gaussian
     * distribution with standard deviation of 0.5.
     * 
     * @see #SimpleNN(int[], Activation[])
     */
    public SimpleNN(int[] shape, Activation[] actvF, Loss lossF, Random rand) {
        this(shape, actvF, lossF);

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
     * This is stochastic training - the network is updated after each
     * input/output training example rather than waiting until the entire
     * training set is run.
     * 
     * @param act An array of input activations for each layer of the network,
     *            where {@code act[0]} is the input to the first layer and
     *            {@code act[act.length-1]} is the output from the last layer.
     * @param err Same structure as {@code act}, but stores the error at each
     *            layer.
     * @param expected An array containing the <b>expected</b> outputs from the
     *            last layer.
     * @param rate The back-propagation rate.
     */
    public void train(float[][] act, float[][] err, float[] expected, float rate) {
        assert (act.length == shape.length && act[0].length == shape[0]);
        assert (err.length == shape.length && err[0].length == shape[0]);
        assert (expected.length == shape[shape.length - 1]);

        // apply NN
        for (int layeridx = 0; layeridx < weights.length; layeridx++) {
            assert (act[layeridx + 1].length == shape[layeridx + 1]);
            assert (err[layeridx + 1].length == shape[layeridx + 1]);

            // compute weighted sum for next layer
            final float[] w = weights[layeridx];
            final float[] v = act[layeridx];
            final float[] nv = err[layeridx + 1];
            final int shapel = shape[layeridx], shapeln = shape[layeridx + 1];
            System.arraycopy(w, 0, nv, 0, shapeln); // bias
            for (int biased_ij = shapeln, i = 0; i < shapel; i++) {
                final float v_i = v[i];
                for (int j = 0; j < shapeln; j++, biased_ij++) {
                    nv[j] += w[biased_ij] * v_i;
                }
            }

            // compute activation function for next layer
            actv[layeridx].activate(nv, act[layeridx + 1]);
        }

        // back-propagation
        final int layermax = shape.length - 1;
        loss.gradient(act[layermax], expected, err[layermax]); // err of last layer
        for (int layeridx = layermax; layeridx > 0; layeridx--) {

            // compute directed gradient of activation function along error vector
            final float[] v = act[layeridx];
            final float[] e = err[layeridx];
            actv[layeridx - 1].dctDerivative(v, e, v);

            // update weights and calculate error for previous layer
            final float[] w = weights[layeridx - 1];
            final float[] pv = act[layeridx - 1];
            final float[] pe = err[layeridx - 1];
            final int shapel = shape[layeridx], shapelp = shape[layeridx - 1];
            for (int j = 0; j < shapel; j++) {
                w[j] -= rate * e[j];    // update bias weights
            }
            for (int biased_ij = shapel, i = 0; i < shapelp; i++) {
                final float pv_i = pv[i];
                float pe_i = 0;
                for (int j = 0; j < shapel; j++, biased_ij++) {
                    pe_i += w[biased_ij] * v[j];
                    w[biased_ij] -= rate * pv_i * e[j];
                }
                pe[i] = pe_i;
            }
        }
    }
}
