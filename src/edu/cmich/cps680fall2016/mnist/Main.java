package edu.cmich.cps680fall2016.mnist;

import java.io.*;

public class Main {

    public static void main(String[] args) throws IOException {
        ImageSet imgf = new ImageSet("data/train-images-idx3-ubyte.gz");
        LabelSet lblf = new LabelSet("data/train-labels-idx1-ubyte.gz");
        byte[] img = null;
        for (int i = 0; imgf.hasNextImage(); i++) {
            img = imgf.nextImage(img);
            byte label = lblf.nextLabel();
            System.out.println(label);
            DispImage b = new DispImage(img, imgf.rowCnt, imgf.colCnt);
            String name = String.format("%05d-%1d.png", i, label);
            b.writePNG(new FileOutputStream(name));
            b.print(System.out);
            if (i >= 10) break;
        }
    }

}
