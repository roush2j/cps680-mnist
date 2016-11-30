package edu.cmich.cps680fall2016.mnist;

import java.awt.*;
import java.awt.event.*;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import javax.swing.*;
import javax.swing.border.EmptyBorder;

/** Image-enabled Console window for better debugging */
public class LogWindow extends PrintStream {

    private final JFrame frame;

    private final JScrollPane pane;

    private final JComponent content;

    /** Create and display a new console window */
    public LogWindow(String title) throws HeadlessException {
        super(new Printer(), true);
        ((Printer) this.out).logwin = this; // lol stupid Java workarounds

        content = Box.createVerticalBox();
        content.setBorder(new EmptyBorder(2, 5, 2, 5));
        pane = new JScrollPane(content);
        pane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        pane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
        pane.getViewport().setBackground(Color.white);
        frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setContentPane(pane);
        frame.setSize(800, 600);
        frame.setVisible(true);
    }

    /** Utility class for capturing output text and displaying in window */
    private static class Printer extends ByteArrayOutputStream {

        private LogWindow logwin;

        @Override public synchronized void write(int b) {
            super.write(b);
            if (b == '\n') appendText();
        }

        @Override public synchronized void write(byte[] b, int off, int len) {
            super.write(b, off, len);
            if (len > 0 && b[off + len - 1] == '\n') appendText();
        }

        private void appendText() {
            String text = toString();
            reset();
            for (String line : text.split("\\n")) {
                // ridiculous hack, AWT labels are single-line only ...
                JLabel lbl = new JLabel(line);
                lbl.setAlignmentY(JComponent.TOP_ALIGNMENT);
                lbl.setFont(Font.decode(Font.MONOSPACED));
                logwin.appendOutput(lbl);
            }
        }
    }

    /** "Print" an image to the console */
    public void println(Image img) {
        JLabel lbl = new JLabel(new ImageIcon(img), JLabel.LEFT);
        lbl.setAlignmentY(JComponent.TOP_ALIGNMENT);
        lbl.setAlignmentX(JComponent.LEFT_ALIGNMENT);
        lbl.setBorder(new EmptyBorder(2, 2, 2, 2));
        appendOutput(lbl);
    }

    /** Append a component to the console output pane */
    private void appendOutput(JComponent output) {
        content.add(output);
        content.revalidate();
        content.repaint();
        JScrollBar vertical = pane.getVerticalScrollBar();
        vertical.setValue(vertical.getMaximum());
    }

    public void anyKeyToClose() {
        JLabel lbl = new JLabel("Press any key to close ...");
        lbl.setAlignmentY(JComponent.TOP_ALIGNMENT);
        lbl.setFont(Font.decode(Font.MONOSPACED));
        lbl.setBackground(Color.lightGray);
        lbl.setOpaque(true);
        appendOutput(lbl);
        frame.addKeyListener(new KeyAdapter() {
            @Override public void keyTyped(KeyEvent e) {
                frame.dispose();
            }
        });
    }
}
