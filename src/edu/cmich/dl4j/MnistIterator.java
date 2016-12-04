package edu.cmich.dl4j;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * An iterator for mnist image and label
 * @author Ogwara O. Rowland
 *
 */
public class MnistIterator implements DataSetIterator{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -4643932469799209747L;
	private MnistDataSet mds;
	
	private int batchSize = 64;

	public MnistIterator(MnistDataSet mds) {
		this(mds, 64);
	}
	
	public MnistIterator(MnistDataSet mds, int batchSize) {
		this.mds = mds;
		this.batchSize = batchSize;
	}

	@Override
	public boolean hasNext() {
		return mds.hasNext();
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

	@Override
	public DataSet next(int num) {
		try {
			MnistImage[] rn = mds.read(num);
			float[][] imgs = new float[rn.length][0];
			float[][] labels = new float[rn.length][0];
			for (int i = 0; i < rn.length; i++) {
				if(rn[i] == null){
					break;
				}
				imgs[i] = rn[i].getImage();
				labels[i] = rn[i].getLabel();
			}
			return new DataSet(Nd4j.create(imgs), Nd4j.create(labels));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}

	@Override
	public int totalExamples() {
		return mds.getImageSize();
	}

	@Override
	public int inputColumns() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int totalOutcomes() {
		return mds.getLabelSize();
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void reset() {
		mds.reset();
		
	}

	@Override
	public int batch() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int cursor() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int numExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException();
		
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		return null;
	}

	@Override
	public List<String> getLabels() {
		return Arrays.asList(new String[]{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});
	}

}
