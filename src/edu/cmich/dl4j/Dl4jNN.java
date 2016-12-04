package edu.cmich.dl4j;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Dl4jNN {
	
	private static Logger log = LoggerFactory.getLogger(Dl4jNN.class);

	public static void main(String[] args) throws FileNotFoundException, IOException {

		MnistIterator trainingSet = new MnistIterator(new MnistDataSet("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"), 512);
		MnistIterator testSet = new MnistIterator(new MnistDataSet("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz"), 512);
		
		final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 10; // number of epochs to perform


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
            .iterations(1)
            .learningRate(0.006) //specify the learning rate
            .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
            .regularization(true).l2(1e-4) // regularize learning model
            .list()
            .layer(0, new DenseLayer.Builder() //create the first input layer.
                    .nIn(numRows * numColumns)
                    .nOut(300)
                    .weightInit(WeightInit.XAVIER)
                    .activation("relu")
                    .build())
//            .layer(1, new DenseLayer.Builder() //create the second input layer
//                    .nIn(300)
//                    .nOut(300)
//                    .activation("relu")
//                    .weightInit(WeightInit.XAVIER)
//                    .build())
            .layer(2, new OutputLayer.Builder(LossFunction.MSE) //create hidden layer
                    .activation("softmax")
                    .nIn(300)
                    .nOut(outputNum)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .pretrain(false).backprop(true) //use backpropagation to adjust weights
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(15));  //print the score with every iteration

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
        	log.info("Epoch " + i);
            model.fit(trainingSet);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(testSet.hasNext()){
            DataSet next = testSet.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
		

	}

}
