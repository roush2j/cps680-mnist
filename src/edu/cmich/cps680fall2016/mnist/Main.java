package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.zip.*;

public class Main {
    

    public static void main(String[] args) throws IOException {
        ImageSet imgf = new ImageSet("data/train-images-idx3-ubyte.gz");
        LabelSet lblf = new LabelSet("data/train-labels-idx1-ubyte.gz");
        for (byte[] img = null; imgf.hasNextImage(); ) {
            img = imgf.nextImage(img);
            byte label = lblf.nextLabel();
            System.out.println(label);
            for (int r = 0; r < imgf.rowCnt; r++) {
                for (int c = 0; c < imgf.colCnt; c++) {
                    int pix = img[imgf.idx(r, c)] & 0xFF;
                    System.out.format("%02X ", pix);
                }
                System.out.println();
            }
        }
    }

}
