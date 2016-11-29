package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

import static edu.cmich.cps680fall2016.mnist.SimpleNN.*;

public class Main {

    public static void main(String[] args) throws IOException {
        SimpleNN nn = new SimpleNN(new int[] { 28 * 28, 10 }, 
                new Activation[] { SOFTMAX }, new Random());

        ImageSet imgf = new ImageSet("data/train-images-idx3-ubyte.gz");
        LabelSet lblf = new LabelSet("data/train-labels-idx1-ubyte.gz");
        float[][] vals = nn.valueArray();
        for (int i = 0; imgf.hasNextImage(); i++) {
            if (i % 1000 == 0) System.out.println("Training on image " + i);
            if (i > 0) break;

            imgf.nextImage(vals[0]);
            byte label = lblf.nextLabel();
            System.out.println(label);
            DispImage b = new DispImage(DispImage.floatPix(vals[0]),
                    imgf.rowCnt, imgf.colCnt);
            String name = String.format("%05d-%1d.png", i, label);
            b.writePNG(new FileOutputStream(name));
            b.print(System.out);

            nn.apply(vals);
            System.out.println(Arrays.toString(vals[1]));
            nn.dump();
        }
    }

}
