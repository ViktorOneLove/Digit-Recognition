package recognition;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

public class FileReader {
    public String[] extractFiles(String folderName){
        File folder = new File(folderName);
        return folder.list();
    }

    /**
     *
     * @param folderName can be null, if file exists in home directory
     */
    public ArrayList<Vector> readFiles(List<String> files, String folderName){
        ArrayList<Vector> digits = new ArrayList<>();
        for(String file : files) {
            digits.add(readFile(file, folderName));
        }
        return digits;
    }

    /**
     *
     * @param folderName can be null, if file exists in home directory
     */
    public Vector readFile(String file, String folderName){
        Vector digits = null;
        try {
            Path path = folderName == null ? Paths.get(file) : Paths.get(folderName, file);
            String[] readData = new String(Files.readAllBytes(path)).split("\\s+");
            digits = new Vector(readData.length);
            int i;
            for (i = 0; i < readData.length - 1; i++) {
                digits.setValue(i, Double.parseDouble(readData[i]) / 255d);
            }
            //last value is an ideal value of number to recognize
            digits.setValue(i, Double.parseDouble(readData[i]));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return digits;
    }
}
