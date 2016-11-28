package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.NoSuchElementException;
import java.util.zip.GZIPInputStream;

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
        this(new GZIPInputStream(new FileInputStream(mnistImageFile), 4096));
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
            throw new IOException("Invalid magic header: input stream does not "
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
     * Return the next image in the set, as an array of {@link #rowCnt *
     * #colCnt} pixels, one byte per pixel, in row-major order.
     */
    public byte[] nextImage(byte[] out) throws IOException {
        if (readCnt >= imageCnt) throw new NoSuchElementException();
        if (out == null) out = new byte[rowCnt * colCnt];
        in.readFully(out, 0, rowCnt * colCnt);
        readCnt++;
        return out;
    }

    public byte[] nextImage() throws IOException {
        return nextImage(null);
    }

    /** Return the 0-based array offset of the pixel at (r,c) */
    public int idx(int r, int c) {
        assert (r >= 0 && r < rowCnt);
        assert (c >= 0 && c < colCnt);
        return r * colCnt + c;
    }
}
