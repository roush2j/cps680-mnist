package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.*;
import static edu.cmich.cps680fall2016.mnist.Activation.*;
import static edu.cmich.cps680fall2016.mnist.Loss.*;
import static edu.cmich.cps680fall2016.mnist.LogWindow.*;

public class Main {

    public static LogWindow out = new LogWindow("Output Log");

    public static void main(String[] args) throws IOException {
        int[] shape = { 28 * 28, 10 };
        int[] widths = { 28, 10 };
        Activation[] actv = { PASSTHROUGH };
        Loss loss = SOFTMAX_CROSS_ENTROPY;
        SimpleNN nn = new SimpleNN(shape, actv, loss, new Random());

        out.printhr("Training ...");
        train(nn, 60000, 0.05F);

        out.printhr("Trained Weights ...");
        printWeights(nn, widths, 10);

        out.printhr("Example Tests ...");
        printTests(nn, widths, 10);

        out.printhr("Testing ...");
        float results = test(nn, 10000);
        out.format("Error Rate: %6.2f%% incorrect\n", (1 - results) * 100F);

        out.anyKeyToClose();
    }

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
                //
                byte label = lbl.nextLabel();
                exp[label] = 1;
                img.nextImage(act[0]);
                //
                nn.train(act, err, exp, rate);
                exp[label] = 0;
            }
        }
    }

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

    public static float test(SimpleNN nn, int count) throws IOException {
        float[][] act = nn.valueArray();
        int correctcnt = 0;

        for (int c = 0; c < count; c++) {
            ImageSet img = new ImageSet("data/t10k-images-idx3-ubyte.gz");
            LabelSet lbl = new LabelSet("data/t10k-labels-idx1-ubyte.gz");
            for (; img.hasNextImage() && lbl.hasNextLabel() && c < count; c++) {
                if (c % 5000 == 0) out.format("Testing image %8d ...\n", c);
                //
                byte label = lbl.nextLabel();
                img.nextImage(act[0]);
                //
                nn.apply(act);
                float[] output = act[act.length - 1];
                int answer = label;
                for (int i = 0; i < output.length; i++) {
                    if (i == label) continue;
                    if (output[i] >= output[answer]) answer = i;
                }
                if (answer == label) correctcnt++;
            }
        }

        return correctcnt / (float) count;
    }

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
                float[] output = act[act.length - 1];
                int answer = label;
                for (int i = 0; i < output.length; i++) {
                    if (i == label) continue;
                    if (output[i] >= output[answer]) answer = i;
                }
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
}
