package recognition;

import java.io.Serializable;

public class Matrix implements Serializable {

    Matrix(int numOfRows){
        value = new double[numOfRows][];
    }

    Matrix(int numOfRows, int numOfColumns){
        value = new double[numOfRows][numOfColumns];
    }

    Matrix(double[][] value){
        this.value = value;
    }

    public void setRow(int idx, Vector row){
        value[idx] = row.asArray();
    }

    public Vector multiply(Vector vector){
        double[] result = new double[value.length];

        for (int row = 0; row < value.length; row++) {
            double sum = 0;
            for (int column = 0; column < value[row].length; column++) {
                sum += value[row][column]
                        * vector.get(column);
            }
            result[row] = sum;
        }
        return new Vector(result);
    }

    public Matrix transpose(){
        double res[][] = new double[value[0].length][];
        for(int i = 0; i < value[0].length; i++){
            res[i] = new double[value.length];
            for(int j = 0; j < value.length; j++){
                res[i][j] = value[j][i];
            }
        }
        return new Matrix(res);
    }

    public void setValue(int rowIdx, int columnIdx, double value){
        this.value[rowIdx][columnIdx] = value;
    }

    public Vector getRow(int idx){
        return new Vector(value[idx]);
    }

    public int length(){
        return value.length;
    }

    private double[][] value;

}
