package edu.cmich.cps680fall2016.mnist;

import java.io.*;
import java.util.NoSuchElementException;
import java.util.zip.GZIPInputStream;

/**
 * A MNIST image label set, parsed from a binary stream.
 */
public class LabelSet {

    public final int labelCnt;

    private int readCnt = 0;

    private final DataInputStream in;

    /**
     * Create a new "stream" of labels.
     * 
     * @param mnistLabelFile The name of a gz-compressed an MNIST label file
     * @throws IOException
     */
    public LabelSet(String mnistLabelFile) throws IOException {
        this(new GZIPInputStream(fileStream(mnistLabelFile)));
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
     * Create a new "stream" of labels.
     * 
     * @param mnistLabelStream An input stream providing raw MNIST label data
     * @throws IOException
     */
    public LabelSet(InputStream mnistLabelStream) throws IOException {
        this.in = new DataInputStream(mnistLabelStream);
        int magic = in.readInt();
        if (magic != 2049)
            throw new IOException(
                    "Invalid magic header: input stream does not "
                            + "appear to be a valid MNIST label set");
        this.labelCnt = in.readInt();
    }

    /** Return true if there is at least one more label in the set */
    public boolean hasNextLabel() {
        return readCnt < labelCnt;
    }

    /** Return the next label in the set, as byte value 0-9. */
    public byte nextLabel() throws IOException {
        if (readCnt >= labelCnt) throw new NoSuchElementException();
        byte val = in.readByte();
        readCnt++;
        return val;
    }
}
