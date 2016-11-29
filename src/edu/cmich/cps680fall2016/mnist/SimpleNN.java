package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class SimpleNN {

    /** The number of inputs/neurons in each layer */
    private final int[] shape;

    /**
     * Weight matrices for each successive pair of layers (0->1, 1->2, ...).
     * Each element {@code i} has size {@code #shape[i] * #shape[i+1]},
     * containing weights in {@code [i]-major} order.
     */
    private final float[][] weights;

    /**
     * Create a new NN with all weights initialized to 0.
     * 
     * @param shape An array containing the size of each layer in the NN.
     */
    public SimpleNN(int[] shape) {
        assert (shape.length >= 2);
        this.shape = shape.clone();
        this.weights = new float[shape.length - 1][];
        for (int layeridx = 0; layeridx < shape.length - 1; layeridx++) {
            weights[layeridx] = new float[shape[layeridx] * shape[layeridx + 1]];
        }
    }

    /**
     * Create a new NN with all weights initialized using a Gaussian
     * distribution with standard deviation of 0.5.
     * 
     * @param shape An array containing the size of each layer in the NN.
     */
    public SimpleNN(int[] shape, Random rand) {
        this(shape);

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

        // apply NN
        for (int layeridx = 0; layeridx < weights.length; layeridx++) {
            assert (values[layeridx + 1].length == shape[layeridx + 1]);

            // compute weighted sum for next layer
            final float[] w = weights[layeridx];
            final float[] v = values[layeridx + 1];
            Arrays.fill(v, 1); // bias
            for (int i = 0, widx = 0; i < shape[0]; i++) {
                final float v_i = values[layeridx][i];
                for (int j = 0; j < shape[layeridx + 1]; j++, widx++) {
                    v[j] += w[widx] * v_i;
                }
            }

            // compute activation function for next layer
            // ... TODO
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

}
