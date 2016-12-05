package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.NoSuchElementException;
import java.util.zip.GZIPInputStream;

/**
 * A MNIST image set, parsed from a binary stream.
 */
public class ImageSet {

    public final int imageCnt, rowCnt, colCnt;

    private int readCnt = 0;

    private final DataInputStream in;

    /**
     * Create a new "stream" of images.
     * 
     * @param mnistImageFile The name of a gz-compressed an MNIST image file
     * @throws IOException
     */
    public ImageSet(String mnistImageFile) throws IOException {
        this(new GZIPInputStream(fileStream(mnistImageFile), 4096));
    }

    private static InputStream fileStream(String filename)
            throws FileNotFoundException {
        File file = new File(filename);
        if (file.exists()) return new FileInputStream(file);
        InputStream stream = ImageSet.class.getResourceAsStream("/" + filename);
        if (stream == null) throw new FileNotFoundException(filename);
        else return stream;
    }

    /**
     * Create a new "stream" of images.
     * 
     * @param mnistImageStream An input stream providing raw MNIST image data
     * @throws IOException
     */
    public ImageSet(InputStream mnistImageStream) throws IOException {
        this.in = new DataInputStream(mnistImageStream);
        int magic = in.readInt();
        if (magic != 2051)
            throw new IOException(
                    "Invalid magic header: input stream does not "
                            + "appear to be a valid MNIST image set");
        this.imageCnt = in.readInt();
        this.rowCnt = in.readInt();
        this.colCnt = in.readInt();
    }

    /** Return true if there is at least one more image in the set */
    public boolean hasNextImage() {
        return readCnt < imageCnt;
    }

    /**
     * Return the next image, as an array of {@code rowCnt*colCnt} pixels, one
     * unsigned byte per pixel, in row-major order.
     */
    public byte[] nextImage(byte[] out) throws IOException {
        if (readCnt >= imageCnt) throw new NoSuchElementException();
        if (out == null) out = new byte[rowCnt * colCnt];
        in.readFully(out, 0, rowCnt * colCnt);
        readCnt++;
        return out;
    }

    /**
     * Return the next image, as an array of {@code rowCnt*colCnt} pixels, one
     * float in the range [0-1] per pixel, in row-major order.
     */
    public float[] nextImage(float[] out) throws IOException {
        if (readCnt >= imageCnt) throw new NoSuchElementException();
        if (out == null) out = new float[rowCnt * colCnt];
        for (int i = 0; i < rowCnt * colCnt; i++) {
            out[i] = in.readUnsignedByte() / 255F;
        }
        readCnt++;
        return out;
    }

    /** Return the 0-based array offset of the pixel at (r,c) */
    public int idx(int r, int c) {
        assert (r >= 0 && r < rowCnt);
        assert (c >= 0 && c < colCnt);
        return r * colCnt + c;
    }
}
