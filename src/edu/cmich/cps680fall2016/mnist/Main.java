package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import static edu.cmich.cps680fall2016.mnist.SimpleNN.*;

public class Main {

    public static LogWindow out = new LogWindow("Test Log!");

    public static void main(String[] args) throws IOException {
        SimpleNN nn = new SimpleNN(new int[] { 28 * 28, 300, 10 },
                new Activation[] { LOGISTIC, LOGISTIC }, new Random());

        float[][] act = nn.valueArray();
        float[][] err = nn.valueArray();
        float[] exp = new float[10];

        for (int rep = 3; rep > 0; rep--) {
            ImageSet imgf = new ImageSet("data/train-images-idx3-ubyte.gz");
            LabelSet lblf = new LabelSet("data/train-labels-idx1-ubyte.gz");
            for (int i = 0; imgf.hasNextImage(); i++) {
                if (i % 1000 == 0) out.println("Training on image " + i);
                //            if (i > 10000) break;

                byte label = lblf.nextLabel();
                Arrays.fill(exp, 0);
                exp[label] = 1;
                //            out.println(label);

                imgf.nextImage(act[0]);
                //            DispImage b = new DispImage(DispImage.floatPix(act[0]),
                //                    imgf.rowCnt, imgf.colCnt);
                //            out.println(b.scaled(8));

                nn.train(act, err, exp, 0.1F);
                //            out.println(new DispImage(DispImage.floatPix(vals[1]), 2, 15).scaled(8));
                //            out.println(new DispImage(DispImage.floatPix(vals[1]), 1, 10).scaled(8));
            }
        }

        //        out.println("Done training ...");
        //        for (int w = 0; w < act.length - 1; w++) {
        //            out.println(nn.weightImage(w));
        //        }

        ImageSet imgt = new ImageSet("data/t10k-images-idx3-ubyte.gz");
        LabelSet lblt = new LabelSet("data/t10k-labels-idx1-ubyte.gz");
        int testcnt = 0, correct = 0;
        for (; imgt.hasNextImage(); testcnt++) {
            if (testcnt % 1000 == 0)
                out.println("Testing on image " + testcnt);
            byte label = lblt.nextLabel();
            imgt.nextImage(act[0]);
            nn.apply(act);
            boolean c = true;
            for (int i = 0; i < 10; i++) {
                if (i == label) continue;
                if (act[act.length - 1][i] >= act[act.length - 1][label])
                    c = false;
            }
            if (c) correct++;
        }
        out.format("Test Results: %6.2f%% correct\n", (correct * 100F)
                / testcnt);
    }

}
