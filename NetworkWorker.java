package recognition;

import java.io.*;
import java.util.*;

public class NetworkWorker {
    public void execute(Scanner scanner) {
        int numOfOperation = Integer.parseInt(scanner.nextLine());
        switch (numOfOperation) {
            case 1:
                long startTime = System.currentTimeMillis();
                doLearning(10, 200);
                long endTime = System.currentTimeMillis();
                System.out.println("Total execution time: " + (endTime-startTime) + "ms");
                break;
            case 2:
                if (!new File(configFileName).exists()) {
                    doLearning(1, 200);
                }
                doAnalysis();
                break;
            case 3:
                if (!new File(configFileName).exists()) {
                    doLearning(1, 200);
                }
                doRecognizing(scanner);
                break;
        }
    }

    private void doLearning(int numOfIterations, int batchSize){
        int[] sizeOfNeuronLayers = new int[]{784, 16, 16, 10};
        NetworkParameters networkParameters = new NetworkParameters();

        NetworkOperations networkOperations = new NetworkOperations();
        networkOperations.setNetworkParameters(networkParameters, sizeOfNeuronLayers);

        FileReader fileReader = new FileReader();
        for(int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++){
            List<String> trainingFiles = Arrays.asList(fileReader.extractFiles(trainingPathName));
            Collections.shuffle(trainingFiles);
            int numOfChunks = (int) Math.ceil((double)trainingFiles.size() / batchSize);
            for (int numOfBatch = 0; numOfBatch < numOfChunks;
                 numOfBatch++){
                int fromIdx = numOfBatch * batchSize;
                int toIdx = Math.min((numOfBatch + 1) * batchSize - 1, trainingFiles.size() - 1);
                ArrayList<Vector> trainingSample = fileReader.readFiles(trainingFiles.subList(fromIdx, toIdx),
                        trainingPathName);
                networkOperations.learn(trainingSample, 3d);
            }
        }

        saveNetworkParameters(networkParameters);
    }

    void doRecognizing(Scanner scanner){
        NetworkParameters networkParameters = loadNetworkParameters();

        NetworkOperations networkOperations = new NetworkOperations();
        networkOperations.loadNetworkParameters(networkParameters);

        String fileName = scanner.nextLine();
        FileReader fileReader = new FileReader();
        Vector numberToRecognize = fileReader.readFile(fileName, null);
        System.out.printf("%d", networkOperations.recognize(numberToRecognize));
    }

    private void doAnalysis(){
        NetworkParameters networkParameters = loadNetworkParameters();

        NetworkOperations networkOperations = new NetworkOperations();
        networkOperations.loadNetworkParameters(networkParameters);

        FileReader fileReader = new FileReader();
        String[] testFiles = fileReader.extractFiles(testPathName);
        int numOfCorrectlyRecognizedNumbers = 0;
        int numAllNumbers = testFiles.length;
        for(String testFile : testFiles) {
            Vector testSample = fileReader.readFile(testPathName, testFile);
            int recognizedNumber = networkOperations.recognize(testSample);
            if (recognizedNumber == testSample.get(testSample.length() - 1)) {
                numOfCorrectlyRecognizedNumbers++;
            }
        }
        System.out.println((double) numOfCorrectlyRecognizedNumbers / numAllNumbers * 100d);
    }

    private void saveNetworkParameters(NetworkParameters networkParameters){
        try {
            FileOutputStream fos = new FileOutputStream(configFileName);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(networkParameters);
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private NetworkParameters loadNetworkParameters(){
        NetworkParameters networkParameters = null;
        try {
            FileInputStream fis = new FileInputStream(configFileName);
            BufferedInputStream bis = new BufferedInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(bis);
            networkParameters = (NetworkParameters)ois.readObject();
            ois.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return networkParameters;
    }

    private final String configFileName = "networkParameters";

    private final String trainingPathName = "trainingData";

    private final String testPathName = "testData";

}
