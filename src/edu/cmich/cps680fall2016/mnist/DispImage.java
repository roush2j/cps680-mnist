package edu.cmich.cps680fall2016.mnist;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import javax.imageio.ImageIO;

/**
 * An 8-bit grayscale image for display or output.
 */
public final class DispImage {

    private final BufferedImage img;

    /**
     * Create an image from an array of bytes (interpreted as unsigned bytes).
     * 
     * @param pixels Pixel color values, in row-major order with the
     *            upper-left-most pixel first.
     * @param rowCnt Number of rows (height).
     * @param colCnt Number of columns (width).
     */
    public DispImage(byte[] pixels, int rowCnt, int colCnt) {
        img = new BufferedImage(colCnt, rowCnt, BufferedImage.TYPE_BYTE_GRAY);
        DataBufferByte buf = (DataBufferByte) img.getRaster().getDataBuffer();
        System.arraycopy(pixels, 0, buf.getData(), 0, rowCnt * colCnt);
    }

    /**
     * Create an image from an array of floats in the range {@code 0.0 - 1.0},
     * inclusive.
     * 
     * @param pixels Pixel color values, in row-major order with the
     *            upper-left-most pixel first.
     * @param rowCnt Number of rows (height).
     * @param colCnt Number of columns (width).
     */
    public DispImage(float[] pixels, int rowCnt, int colCnt) {
        img = new BufferedImage(colCnt, rowCnt, BufferedImage.TYPE_BYTE_GRAY);
        DataBufferByte buf = (DataBufferByte) img.getRaster().getDataBuffer();
        byte[] rawbuf = buf.getData();
        for (int i = 0; i < rowCnt * colCnt; i++) {
            rawbuf[i] = (byte) (pixels[i] * 255);
        }
    }

    /** Write this image to {@code out} in PNG format. */
    public void writePNG(OutputStream out) throws IOException {
        ImageIO.write(img, "png", out);
    }
    
    /** Write this image to {@code out} as space-separated hex bytes. */
    public void print(PrintStream out) {
        for (int r = 0; r < img.getHeight(); r++) {
            for (int c = 0; c < img.getWidth(); c++) {
                // technically this is just the G component, but R&B should be same
                int pix = img.getRGB(c, r) & 0xFF;
                System.out.format("%02X ", pix);
            }
            System.out.println();
        }
    }
}
