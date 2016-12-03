package edu.cmich.cps680fall2016.mnist;

import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import javax.swing.*;
import javax.swing.border.*;

/** Image-enabled Console window for better debugging */
public class LogWindow extends PrintStream {

    private final JFrame frame;

    private final JComponent content;

    private final Printer out;

    /** Create and display a new console window */
    public LogWindow(String title) throws HeadlessException {
        super(new Printer(), true);
        this.out = (Printer) super.out;
        this.out.logwin = this; // lol stupid Java workarounds

        content = Box.createVerticalBox();
        content.setBorder(new EmptyBorder(2, 5, 2, 5));
        JScrollPane pane = new JScrollPane(content);
        pane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        pane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
        pane.getViewport().setBackground(Color.white);
        frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setContentPane(pane);
        frame.setSize(800, 600);
        frame.setVisible(true);
    }

    /** Utility class for capturing output text and displaying in window */
    private static class Printer extends ByteArrayOutputStream {

        private LogWindow logwin;

        @Override public synchronized void write(int b) {
            super.write(b);
            if (b == '\n') flushText();
        }

        @Override public synchronized void write(byte[] b, int off, int len) {
            super.write(b, off, len);
            if (len > 0 && b[off + len - 1] == '\n') flushText();
        }

        public void flushText() {
            String text = toString();
            reset();
            // ridiculous hack, AWT labels are single-line only ...
            for (String line : text.split("\\n")) {
                logwin.appendOutput(txtC(line));
            }
        }
    }

    /** Append a component to the console output pane */
    private void appendOutput(JComponent output) {
        if (output.getMaximumSize().height > output.getPreferredSize().height) {
            int w = output.getMaximumSize().width;
            int h = output.getPreferredSize().height;
            output.setMaximumSize(new Dimension(w, h));
        }
        content.add(output);
        content.revalidate();
        content.repaint();
        SwingUtilities.invokeLater(new Runnable() {
            @Override public void run() {
                output.scrollRectToVisible(output.getBounds());
            }
        });
    }

    /** "Print" an arbitrary Swing component */
    public void println(JComponent output) {
        out.flushText();
        output.setAlignmentX(JComponent.LEFT_ALIGNMENT);
        appendOutput(output);
    }

    /** Build an printable component from an image */
    public static JComponent imgC(Image img) {
        JLabel imglbl = new JLabel(new ImageIcon(img));
        Border b = new LineBorder(Color.black);
        Border margin = new EmptyBorder(2, 2, 2, 2);
        imglbl.setBorder(new CompoundBorder(margin, b));
        return imglbl;
    }

    /** Build an printable component from a string */
    public static JComponent txtC(String text) {
        JLabel lbl = new JLabel(text);
        lbl.setFont(Font.decode(Font.MONOSPACED));
        return lbl;
    }

    /** Build an printable component from a string of the given RGB color */
    public static JComponent txtC(String text, int color) {
        JComponent lbl = txtC(text);
        lbl.setForeground(new Color(color));
        return lbl;
    }

    /**
     * Build a printable horizontal group of strings, images, or nested
     * components
     */
    public static JComponent hgrpC(Object... items) {
        Box box = Box.createHorizontalBox();
        for (Object item : items) {
            JComponent cmp;
            if (item instanceof JComponent) {
                cmp = (JComponent) item;
            } else if (item instanceof Image) {
                cmp = imgC((Image) item);
            } else {
                cmp = txtC(item.toString());
            }
            cmp.setAlignmentY(JComponent.CENTER_ALIGNMENT);
            box.add(cmp);
            box.add(Box.createHorizontalStrut(5));
        }
        return box;
    }

    /**
     * Build a printable vertical group of strings, images, or nested components
     */
    public static JComponent vgrpC(Object... items) {
        Box box = Box.createVerticalBox();
        for (Object item : items) {
            JComponent cmp;
            if (item instanceof JComponent) {
                cmp = (JComponent) item;
            } else if (item instanceof Image) {
                cmp = imgC((Image) item);
            } else {
                cmp = txtC(item.toString());
            }
            cmp.setAlignmentX(JComponent.CENTER_ALIGNMENT);
            box.add(cmp);
        }
        return box;
    }

    /** "Print" a horizontal break */
    public void printhr(String label) {
        JComponent lbl = txtC(label);
        lbl.setFont(lbl.getFont().deriveFont(Font.BOLD));
        int width = Integer.MAX_VALUE; // span entire width of window
        int height = lbl.getMaximumSize().height;
        lbl.setMaximumSize(new Dimension(width, height));
        lbl.setBackground(Color.lightGray);
        lbl.setOpaque(true);
        println(lbl);
    }

    /** Close the window after the next key press. */
    public void anyKeyToClose() {
        printhr("Press any key to close ...");
        frame.addKeyListener(new KeyAdapter() {
            @Override public void keyTyped(KeyEvent e) {
                frame.dispose();
            }
        });
    }
}
