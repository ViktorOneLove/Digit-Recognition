package recognition;

import java.util.*;

public class NetworkOperations {
    public void learn(ArrayList<Vector> trainingData, double eta) {
        ArrayList<Pair<ArrayList<Matrix>, Matrix>> nablas = new ArrayList<>();
        for (int idxOfTrainingSample = 0; idxOfTrainingSample < trainingData.size();
             idxOfTrainingSample++) {
            //store all nablas in array
            nablas.add(backPropagation(trainingData.get(idxOfTrainingSample)));
        }

        //update weights
        for (int layerIdx = 0; layerIdx < networkParameters.numOfWeightLayers; layerIdx++) {
            Matrix weightMatrixOfThisLayer = networkParameters.neuronWeights.get(layerIdx);
            for (int neuronToIdx = 0; neuronToIdx < networkParameters.sizeOfNeuronLayers[layerIdx + 1];
                 neuronToIdx++) {
                for (int neuronFromIdx = 0; neuronFromIdx < networkParameters.sizeOfNeuronLayers[layerIdx];
                     neuronFromIdx++) {
                    double sumOfWeightNablas = 0;
                    for (int numOfSample = 0; numOfSample < nablas.size(); numOfSample++) {
                        Matrix nablasOfThisWeightLayer = nablas.get(numOfSample).getFirst().get(layerIdx);
                        double nablaWeight = nablasOfThisWeightLayer.getRow(neuronToIdx).get(neuronFromIdx);
                        sumOfWeightNablas += nablaWeight;
                    }
                    double newWeight = weightMatrixOfThisLayer.getRow(neuronToIdx).get(neuronFromIdx) -
                            eta / nablas.size() * sumOfWeightNablas;
                    weightMatrixOfThisLayer.setValue(neuronToIdx, neuronFromIdx, newWeight);
                }
            }
            networkParameters.neuronWeights.set(layerIdx, weightMatrixOfThisLayer);
        }

        //update biases
        for (int layerIdx = 0; layerIdx < networkParameters.numOfWeightLayers; layerIdx++) {
            Vector biasesOfThisLayer = networkParameters.biasesWeights.get(layerIdx);
            for (int neuronToIdx = 0; neuronToIdx < networkParameters.sizeOfNeuronLayers[layerIdx + 1];
                 neuronToIdx++) {
                double sumOfBiasesNablas = 0;
                for (int numOfSample = 0; numOfSample < nablas.size(); numOfSample++) {
                    Vector nablasOfThisBiasesLayer = nablas.get(numOfSample).getSecond().getRow(layerIdx);
                    double nablaBias = nablasOfThisBiasesLayer.get(neuronToIdx);
                    sumOfBiasesNablas += nablaBias;
                }
                double newBias = biasesOfThisLayer.get(neuronToIdx) - eta / nablas.size() * sumOfBiasesNablas;
                biasesOfThisLayer.setValue(neuronToIdx, newBias);
            }
            networkParameters.biasesWeights.set(layerIdx, biasesOfThisLayer);
        }
    }

    public int recognize(Vector digitToRecognize){
        Vector lastLayerResult = feedForward(digitToRecognize).getSecond().
                getRow(networkParameters.numOfNeuronLayers - 1);
        return lastLayerResult.findIdxOfMaxValue();
    }

    public void loadNetworkParameters(NetworkParameters networkParameters){
        this.networkParameters = networkParameters;
    }

    public void setNetworkParameters(NetworkParameters networkParameters,
                                     int[] sizeOfNeuronLayers){
        this.networkParameters = networkParameters;

        networkParameters.numOfNeuronLayers = sizeOfNeuronLayers.length;

        networkParameters.numOfWeightLayers = sizeOfNeuronLayers.length - 1;

        networkParameters.sizeOfNeuronLayers = sizeOfNeuronLayers;

        setNeuronWeights();

        setBiasesWeights();

    }


    private void setNeuronWeights(){
        networkParameters.neuronWeights = new ArrayList<>();
        int weightLayerIdx = 0;
        for(int i = 1; i < networkParameters.numOfNeuronLayers; i++){
            Matrix matrix = new Matrix(networkParameters.sizeOfNeuronLayers[i],
                    networkParameters.sizeOfNeuronLayers[i-1]);
            Random rand = new Random();
            for(int j = 0; j < matrix.length(); j++){
                for(int k = 0; k < matrix.getRow(j).length(); k++){
                    matrix.setValue(j, k, rand.nextGaussian());
                }
            }
            networkParameters.neuronWeights.add(weightLayerIdx++, matrix);
        }
    }

    private void setBiasesWeights(){
        networkParameters.biasesWeights = new ArrayList<>();
        for(int biasLayerIdx = 0; biasLayerIdx < networkParameters.numOfWeightLayers; biasLayerIdx++){
            Vector vector = new Vector(networkParameters.sizeOfNeuronLayers[biasLayerIdx + 1]);
            Random rand = new Random();
            for(int i = 0; i < vector.length(); i++){
                vector.setValue(i, rand.nextGaussian());
            }
            networkParameters.biasesWeights.add(biasLayerIdx, vector);
        }
    }



    /**
     *
     * backPropagation algorithm
     * @param trainingNeurons array of training data
     * @return nabla for neuron weights and biases
     */
    private Pair<ArrayList<Matrix>, Matrix> backPropagation(Vector trainingNeurons){
        //first matrix of weighted input
        //second matrix of sigmaOfWeighted input
        Pair<Matrix, Matrix> feedRes = feedForward(trainingNeurons);
        Matrix weightedInput = feedRes.getFirst();
        Matrix sigmaOfWeightedInput = feedRes.getSecond();

        //matrix of errors in every neuron
        Matrix errors = new Matrix(networkParameters.numOfNeuronLayers);

        //first, set the last layer error
        //in MNIST data set the last 785-th value in vector - is ideal number
        Vector idealVector = new Vector(networkParameters.
                sizeOfNeuronLayers[networkParameters.numOfNeuronLayers - 1]);
        idealVector.fillWith(0);
        idealVector.setValue((int)trainingNeurons.get(trainingNeurons.length() - 1),
                1);

        Vector derivativeOfSigmoid = sigmoidDerivative(weightedInput.
                getRow(networkParameters.numOfNeuronLayers - 1));
        Vector lastLayerErrors = sigmaOfWeightedInput.
                getRow(networkParameters.numOfNeuronLayers - 1).
                subtraction(idealVector).hadamardProduct(derivativeOfSigmoid);
        errors.setRow(errors.length() - 1, lastLayerErrors);

        //second, find errors in remaining layers
        //iterate from last layer to second, as we don't need to count errors in the first(input) layer
        for(int layerIdx = networkParameters.numOfNeuronLayers - 2; layerIdx >= 1;
            layerIdx--){
            Vector derivativeOfSigmoidLayer = sigmoidDerivative(weightedInput.getRow(layerIdx));
            Matrix transposeMatrixOfWeights = networkParameters.neuronWeights.get(layerIdx).transpose();
            Vector errorsInPrevLayer = errors.getRow(layerIdx + 1);

            Vector errorsInLayer = transposeMatrixOfWeights.multiply(errorsInPrevLayer).
                    hadamardProduct(derivativeOfSigmoidLayer);
            errors.setRow(layerIdx, errorsInLayer);
        }

        //third, count nablas
        ArrayList<Matrix> nablaWeights = new ArrayList<>();
        Matrix nablaBiases = new Matrix(networkParameters.numOfWeightLayers);

        for(int layerIdx = 0; layerIdx < networkParameters.numOfWeightLayers;
            layerIdx++){
            nablaBiases.setRow(layerIdx, errors.getRow(layerIdx + 1));
            Matrix nablaWeight = errors.getRow(layerIdx + 1).multiply(sigmaOfWeightedInput.getRow(layerIdx));
            nablaWeights.add(nablaWeight);
        }

        return new Pair<>(nablaWeights, nablaBiases);
    }


    /**
     *
     * @return weighted input and sigma function of this weighted input
     */
    private Pair<Matrix, Matrix> feedForward(Vector trainingNeurons) {
        Matrix weightedInput = new Matrix(networkParameters.numOfNeuronLayers);
        Matrix sigmaOfWeightedInput = new Matrix(networkParameters.numOfNeuronLayers);

        weightedInput.setRow(0, trainingNeurons);
        sigmaOfWeightedInput.setRow(0, trainingNeurons);

        for(int idxOfLayer = 0; idxOfLayer < networkParameters.numOfNeuronLayers - 1;
            idxOfLayer++){

            weightedInput.setRow(idxOfLayer + 1, networkParameters.neuronWeights.
                    get(idxOfLayer).multiply(sigmaOfWeightedInput.getRow(idxOfLayer)).
                    sum(networkParameters.biasesWeights.get(idxOfLayer)));

            sigmaOfWeightedInput.setRow(idxOfLayer + 1,
                    sigmoidFunction(weightedInput.getRow(idxOfLayer + 1)));

        }

        return new Pair<>(weightedInput, sigmaOfWeightedInput);
    }


    private Vector sigmoidFunction(Vector vector){
        Vector res = new Vector(vector.length());
        for(int i = 0; i < res.length(); i++){
            res.setValue(i, 1 / (1 + Math.exp(-vector.get(i))));
        }
        return res;
    }

    private Vector sigmoidDerivative(Vector vector){
        Vector sigmoid = sigmoidFunction(vector);
        Vector unitVector = new Vector(vector.length());
        unitVector.fillWith(1d);

        return sigmoid.hadamardProduct(unitVector.subtraction(sigmoid));
    }

    private NetworkParameters networkParameters;

}
