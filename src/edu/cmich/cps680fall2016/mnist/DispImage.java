package edu.cmich.cps680fall2016.mnist;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import javax.imageio.ImageIO;

/**
 * An 8-bit grayscale image for display or output.
 */
public final class DispImage extends BufferedImage {

    /**
     * Create an image from an array of bytes (interpreted as unsigned bytes).
     * 
     * @param pixels Pixel color values, in row-major order with the
     *            upper-left-most pixel first.
     * @param rowCnt Number of rows (height).
     * @param colCnt Number of columns (width).
     */
    public DispImage(byte[] pixels, int rowCnt, int colCnt) {
        super(colCnt, rowCnt, BufferedImage.TYPE_BYTE_GRAY);
        DataBufferByte buf = (DataBufferByte) getRaster().getDataBuffer();
        System.arraycopy(pixels, 0, buf.getData(), 0, rowCnt * colCnt);
    }

    /**
     * Create an image from an array-like functor returning normalized floats.
     * 
     * @param pixels Pixel color values, in row-major order with the
     *            upper-left-most pixel at 0.
     * @param rowCnt Number of rows (height).
     * @param colCnt Number of columns (width).
     */
    public DispImage(PixelGen pixels, int rowCnt, int colCnt) {
        super(colCnt, rowCnt, BufferedImage.TYPE_BYTE_GRAY);
        DataBufferByte buf = (DataBufferByte) getRaster().getDataBuffer();
        byte[] rawbuf = buf.getData();
        for (int p = 0; p < rowCnt * colCnt; p++) {
            double norm = pixels.getPixel(p);
            rawbuf[p] = (byte) (Math.min(1, Math.max(0, norm)) * 255);
        }
    }

    /** An array-like functor for generating normalized pixel color values */
    @FunctionalInterface public static interface PixelGen {

        public float getPixel(int idx);

        public default PixelGen stride(final int offset, final int stride) {
            final PixelGen base = this;
            return new PixelGen() {

                @Override public float getPixel(int idx) {
                    return base.getPixel(offset + stride * idx);
                }
            };
        }

        public default PixelGen normalize(final float min, final float max) {
            final PixelGen base = this;
            return new PixelGen() {

                @Override public float getPixel(int idx) {
                    float norm = (base.getPixel(idx) - min) / (max - min);
                    return Math.min(max, Math.max(min, norm));
                }
            };
        }

    }

    public static PixelGen floatPix(final float[] pixels) {
        return new PixelGen() {

            @Override public float getPixel(int offset) {
                return pixels[offset];
            }
        };
    }

    /** Write this image to {@code out} in PNG format. */
    public void writePNG(OutputStream out) throws IOException {
        ImageIO.write(this, "png", out);
    }

    /** Return a scaled-up version of this image */
    public Image scaled(int scale) {
        return this.getScaledInstance(getWidth() * scale, getHeight() * scale,
                Image.SCALE_FAST);
    }

    /** Print this image as space-separated hex bytes. */
    @Override public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < getHeight(); r++) {
            if (r > 0) sb.append('\n');
            for (int c = 0; c < getWidth(); c++) {
                // technically this is just the G component, but R&B should be same
                int pix = getRGB(c, r) & 0xFF;
                int c0 = pix & 0xF, c1 = (pix >> 4) & 0xF;
                sb.append((char) (c0 > 9 ? c0 + 'A' : c0 + '0'));
                sb.append((char) (c1 > 9 ? c1 + 'A' : c1 + '0'));
                sb.append(' ');
            }
        }
        return sb.toString();
    }
}
