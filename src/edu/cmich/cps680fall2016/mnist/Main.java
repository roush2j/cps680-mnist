package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import static edu.cmich.cps680fall2016.mnist.Activation.*;

public class Main {

    public static LogWindow out = new LogWindow("Output Log");

    public static void main(String[] args) throws IOException {
        SimpleNN nn = new SimpleNN(new int[] { 28 * 28, 300, 10 },
                new Activation[] { LOGISTIC, SOFTMAX }, new Random());

        out.println("Training ...");
        train(nn, 10000);

        test(nn, 20, true);

        out.println("Testing ...");
        float results = test(nn, 10000, false);
        out.format("Test Errors: %6.2f%% incorrect\n", (1 - results) * 100F);

        out.anyKeyToClose();
    }

    public static void train(SimpleNN nn, int count) throws IOException {
        float[][] act = nn.valueArray();
        float[][] err = nn.valueArray();
        float[] exp = new float[10];
        float rate = 0.1F;

        for (int c = 0; c < count; c++) {
            ImageSet img = new ImageSet("data/train-images-idx3-ubyte.gz");
            LabelSet lbl = new LabelSet("data/train-labels-idx1-ubyte.gz");
            for (; img.hasNextImage() && lbl.hasNextLabel() && c < count; c++) {
                if (c % 5000 == 0) out.format("Training image %8d ...\n", c);
                //
                byte label = lbl.nextLabel();
                Arrays.fill(exp, 0);
                exp[label] = 1;
                img.nextImage(act[0]);
                //
                nn.train(act, err, exp, rate);
            }
        }
    }

    public static float test(SimpleNN nn, int count, boolean print)
            throws IOException {
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
                    if (output[i] >= output[answer]) answer = i;
                }
                if (answer == label) correctcnt++;
                //
                if (print) {
                    out.println(label);
                    for (float[] a : act) {
                        int h = (a.length + img.colCnt - 1) / img.colCnt;
                        int w = a.length / h;
                        DispImage d = new DispImage(DispImage.floatPix(a), h, w);
                        out.println(d.scaled(6));
                    }
                    out.print("Classifier result: " + answer);
                    if (answer != label) out.print(" !! EXPECTED " + label);
                    out.println();
                }
            }
        }

        return correctcnt / (float) count;
    }
}
