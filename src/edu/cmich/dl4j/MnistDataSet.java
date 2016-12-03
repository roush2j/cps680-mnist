package edu.cmich.dl4j;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

public class MnistDataSet {
	
	private DataInputStream imageData;
	private DataInputStream labelData;
	
	private int imageSize;
	private int labelSize;
	
	private int xDimension;
	private int yDimension;
	
	private int imgIndex;
	
	private String data;
	private String label;
	
	public MnistDataSet(String data, String label) {
		this.data = data;
		this.label = label;
		
		init();
	}
	
	private void init(){
		try {
			GZIPInputStream gzip = new GZIPInputStream(new FileInputStream(data));
			imageData = new DataInputStream(gzip);
			
			imageData.readInt(); // magic number
			imageSize = imageData.readInt();
			xDimension = imageData.readInt();
			yDimension = imageData.readInt();
			
			GZIPInputStream lzip = new GZIPInputStream(new FileInputStream(label));
			labelData = new DataInputStream(lzip);
			
			labelData.readInt(); // magic label number
			labelSize = labelData.readInt();
			
			imgIndex = 0;
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void close(){
		try {
			imageData.close();
			labelData.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void reset(){
		close();
		init();
	}
	
	public boolean hasNext(){
		return imgIndex < imageSize;
	}
	
	public MnistImage[] read(int size) throws IOException{
		MnistImage[] imgs = new MnistImage[size];
		for (int i = 0; i < imgs.length; i++) {
			imgs[i] = readNext();
		}
		return imgs;
	}
	
	public int getImageSize() {
		return imageSize;
	}
	
	public int getLabelSize() {
		return labelSize;
	}
	
	public MnistImage readNext() throws IOException{
		if(imgIndex >= imageSize){
			return null;
		}else{
			byte[] imgSrc = new byte[xDimension * yDimension];
			float[] label = new float[10]; //0 ... 9
			imageData.readFully(imgSrc, 0, imgSrc.length);
			float[] img = new float[imgSrc.length];
			for (int i = 0; i < imgSrc.length; i++) {
				img[i] = imgSrc[i]/255f;
			}
			int rb = labelData.readUnsignedByte();
			label[rb] = 1.0f;
			MnistImage mimg = new MnistImage(img, label);
			imgIndex += 1;
			return mimg;
		}
	}

}
