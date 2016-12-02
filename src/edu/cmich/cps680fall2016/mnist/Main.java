package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.*;
import static edu.cmich.cps680fall2016.mnist.Activation.*;
import static edu.cmich.cps680fall2016.mnist.Loss.*;
import static edu.cmich.cps680fall2016.mnist.LogWindow.*;

public class Main {

    public static LogWindow out;

    public static void main(String[] args) throws IOException {

        final int lmax;
        final int[] shape;
        final int[] widths;
        final Activation[] actv;
        final Loss loss;
        final int trainingcnt;
        final float learningrate;

        // parse arguments
        try {
            // number of layers
            Exception error = new RuntimeException();
            lmax = (args.length - 4) / 2;
            if (lmax * 2 + 4 != args.length) throw error;
            if (lmax < 1) throw error;
            // layer sizes
            shape = new int[lmax + 1];
            widths = new int[lmax + 1];
            for (int i = 0; i <= lmax; i++) {
                String[] size = args[2 * i].split("\\*");
                if (size.length != 2) throw error;
                widths[i] = Integer.parseInt(size[0]);
                shape[i] = Integer.parseInt(size[1]) * widths[i];
            }
            if (shape[0] != 28 * 28) throw error;
            if (shape[lmax] != 10) throw error;
            // activation functions
            actv = new Activation[lmax];
            for (int i = 0; i < lmax; i++) {
                String a = args[2 * i + 1];
                if ("pass".equals(a)) actv[i] = PASSTHROUGH;
                else if ("logistic".equals(a)) actv[i] = LOGISTIC;
                else if ("softmax".equals(a)) actv[i] = SOFTMAX;
                else throw error;
            }
            // loss function
            String l = args[2 * lmax + 1];
            if ("mse".equals(l)) loss = MEAN_SQUARED_ERR;
            else if ("cross".equals(l)) loss = CROSS_ENTROPY;
            else if ("softcross".equals(l)) loss = SOFTMAX_CROSS_ENTROPY;
            else throw error;
            // training params
            trainingcnt = Integer.parseInt(args[args.length - 2]);
            learningrate = Float.parseFloat(args[args.length - 1]);
        } catch (Exception e) {
            String[] usage = { "mnist", "INPUT",
                    "[ACTIV1 HIDDEN1 [ACTIV2 HIDDEN2 [ ...]]]", "ACTIVOUT",
                    "OUTPUT", "LOSS", "TRAINCNT", "LRATE" };
            String[] argdesc = { //
                    "INPUT:    size of input                 28*28", //
                    "ACTIV_:   activation function           pass | logistic | softmax", //
                    "HIDDEN_:  size of hidden layer          <width>*<height>", //
                    "OUTPUT:   size of output layer          10*1", //
                    "LOSS:     loss function                 mse | cross | softcross", //
                    "TRAINCNT: number of training examples   <any positive integer>", //
                    "LRATE:    learning rate                 <float between 0 and 1>" //
            };
            System.err.println(String.join(" ", usage));
            System.err.println("    " + String.join("\n    ", argdesc));
            return;
        }

        out = new LogWindow("MNIST Output Log");
        out.printhr("Network Parameters");
        out.format("Input   :%10s %5d\n", "", shape[0]);
        for (int i = 1; i < lmax; i++) {
            out.format("Hidden %d:%10s %5d\n", i, args[2 * i + 1], shape[i]);
        }
        out.format("Output  :%10s %5d\n", args[2 * lmax - 1], shape[lmax]);
        out.format("Loss    :%10s\n", args[2 * lmax + 1]);
        SimpleNN nn = new SimpleNN(shape, actv, loss, new Random());

        out.printhr("Training ...");
        out.format("%d examples with rate = %f\n", trainingcnt, learningrate);
        train(nn, trainingcnt, learningrate);

        out.printhr("Trained Weights ...");
        printWeights(nn, widths, 10);

        out.printhr("Example Tests ...");
        printTests(nn, widths, 10);

        out.printhr("Testing ...");
        float err = test(nn, 10000);
        out.format("Error Rate: %6.2f%% incorrect\n", err * 100F);

        out.anyKeyToClose();
    }

    /**
     * Train a neural network using examples from the MNIST training data.
     * 
     * @param nn The network to train
     * @param count The number of examples to train on
     * @param rate The learning rate
     * @throws IOException if the data files are missing or unreadable
     */
    public static void train(SimpleNN nn, int count, float rate)
            throws IOException {
        float[][] act = nn.valueArray();
        float[][] err = nn.valueArray();
        float[] exp = new float[10];

        for (int c = 0; c < count; c++) {
            ImageSet img = new ImageSet("data/train-images-idx3-ubyte.gz");
            LabelSet lbl = new LabelSet("data/train-labels-idx1-ubyte.gz");
            for (; img.hasNextImage() && lbl.hasNextLabel() && c < count; c++) {
                if (c % 5000 == 0) out.format("Training image %8d ...\n", c);
                byte label = lbl.nextLabel();
                img.nextImage(act[0]);
                //
                exp[label] = 1;
                nn.train(act, err, exp, rate);
                exp[label] = 0;
            }
        }
    }

    /**
     * Print the weight parameters of a neural network.
     * 
     * @param nn The network to train
     * @param widths The width of the weight plots for each layer.
     * @param limit The maximum number of weight plots to show for each layer.
     */
    public static void printWeights(SimpleNN nn, int[] widths, int limit) {
        for (int lidx = 0; lidx < nn.shape.length - 1; lidx++) {
            List<Object> cmp = new ArrayList<>();
            String lblprefix = "[L" + lidx + "] -> L" + (lidx + 1) + "_";
            for (int i = 0; i < Math.min(limit, nn.shape[lidx + 1]); i++) {
                DispImage img = new DispImage( //
                        DispImage.floatPix(nn.weights[lidx]) //
                                .normalize(-1, 1) //
                                .stride(i, nn.shape[lidx + 1]), //
                        nn.shape[lidx] / widths[lidx], widths[lidx]);
                cmp.add(vgrpC(img.scaled(4), lblprefix + i));
            }
            int rem = nn.shape[lidx + 1] - limit;
            if (rem > 0) cmp.add(rem + " more ...");
            out.println(hgrpC(cmp.toArray()));
        }
    }

    /**
     * Test a neural network using examples from the MNIST testing data.
     * 
     * @param nn The network to train
     * @param count The number of examples to test
     * @return The error rate (fraction of incorrectly classified examples)
     * @throws IOException if the data files are missing or unreadable
     */
    public static float test(SimpleNN nn, int count) throws IOException {
        float[][] act = nn.valueArray();
        int errcnt = 0;

        for (int c = 0; c < count; c++) {
            ImageSet img = new ImageSet("data/t10k-images-idx3-ubyte.gz");
            LabelSet lbl = new LabelSet("data/t10k-labels-idx1-ubyte.gz");
            for (; img.hasNextImage() && lbl.hasNextLabel() && c < count; c++) {
                if (c % 5000 == 0) out.format("Testing image %8d ...\n", c);
                byte label = lbl.nextLabel();
                img.nextImage(act[0]);
                //
                nn.apply(act);
                int answer = maxidx(act[act.length - 1]);
                if (answer != label) errcnt++;
            }
        }

        return errcnt / (float) count;
    }

    /**
     * Print example test results for a neural network using examples from the
     * MNIST testing data.
     * 
     * @param nn The network to train
     * @param widths The width of the activation plot for each layer.
     * @param count The number of examples to test
     * @throws IOException if the data files are missing or unreadable
     */
    public static void printTests(SimpleNN nn, int[] widths, int count)
            throws IOException {
        float[][] act = nn.valueArray();
        float[] exp = new float[10];

        for (int c = 0; c < count; c++) {
            ImageSet img = new ImageSet("data/t10k-images-idx3-ubyte.gz");
            LabelSet lbl = new LabelSet("data/t10k-labels-idx1-ubyte.gz");
            for (; img.hasNextImage() && lbl.hasNextLabel() && c < count; c++) {
                byte label = lbl.nextLabel();
                img.nextImage(act[0]);
                //
                exp[label] = 1;
                float loss = nn.test(act, exp);
                exp[label] = 0;
                int answer = maxidx(act[act.length - 1]);
                //
                List<Object> cmp = new ArrayList<>();
                for (int i = 0; i < act.length; i++) {
                    DispImage im = new DispImage( //
                            DispImage.floatPix(act[i]), //
                            act[i].length / widths[i], widths[i]);
                    cmp.add(vgrpC(im.scaled(4), "L" + i));
                }
                Object clbl = "Classifier result: " + answer;
                if (answer != label) {
                    clbl = vgrpC(clbl, txtC("EXPECTED " + label, 0xE00000));
                }
                cmp.add(vgrpC(clbl, "Loss: " + loss));
                out.println(hgrpC(cmp.toArray()));
            }
        }
    }

    /**
     * Return the index of the (last) maximum value, or -1 if any values are NaN
     */
    private static int maxidx(float[] vals) {
        int idx = -1;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vals.length; i++) {
            if (Float.isNaN(vals[i])) return -1;
            else if (vals[i] >= max) max = vals[idx = i];
        }
        return idx;
    }
}
