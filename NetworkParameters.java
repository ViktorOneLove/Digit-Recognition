package recognition;

import java.io.Serializable;
import java.util.ArrayList;

public class NetworkParameters implements Serializable {
    private static final long serialVersionUID = 191241L;

    int numOfNeuronLayers;

    int numOfWeightLayers;

    int[] sizeOfNeuronLayers;

    ArrayList<Matrix> neuronWeights;

    ArrayList<Vector> biasesWeights;

}
