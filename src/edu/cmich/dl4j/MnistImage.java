package edu.cmich.dl4j;

import java.util.Arrays;

public class MnistImage {
	
	private float[] img;
	private float[] label;
	
	public MnistImage(float[] imgSrc, float[] label) {
		this.img = imgSrc;
		this.label = label;
	}
	
	public float[] getImage() {
		return this.img;
	}
	
	public float[] getLabel() {
		return label;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(img) + " label: " + Arrays.toString(label);
	}

}
