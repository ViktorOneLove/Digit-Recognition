package recognition;

import java.io.Serializable;
import java.util.Arrays;

public class Vector implements Serializable {

    Vector(int size){
        value = new double[size];
    }

    Vector(double[] value){
        this.value = value;
    }

    public Vector sum(Vector vector){
        double[] res = new double[vector.length()];
        for(int i = 0; i < res.length; i++){
            res[i] = value[i] + vector.get(i);
        }
        return new Vector(res);
    }

    public Vector subtraction(Vector vector){
        double[] res = new double[vector.length()];
        for(int i = 0; i < res.length; i++){
            res[i] = value[i] - vector.get(i);
        }
        return new Vector(res);
    }

    public Matrix multiply(Vector vector){
        double[][] res = new double[value.length][vector.length()];
        for(int i = 0; i < value.length; i++){
            for(int j = 0; j < vector.length(); j++){
                res[i][j] = value[i] * vector.get(j);
            }
        }
        return new Matrix(res);
    }

    public Vector hadamardProduct(Vector vector){
        double[] res = new double[vector.length()];
        for(int i = 0; i < vector.length(); i++){
            res[i] = value[i] * vector.get(i);
        }
        return new Vector(res);
    }

    public int findIdxOfMaxValue(){
        int maxValueIdx = 0;
        for(int i = 0; i < value.length; i++){
            if(value[i] > value[maxValueIdx])
                maxValueIdx = i;
        }
        return maxValueIdx;
    }

    public void fillWith(double value){
        Arrays.fill(this.value, value);
    }

    public double get(int idx){
        return value[idx];
    }

    public void setValue(int idx, double value){
        this.value[idx] = value;
    }

    public int length(){
        return value.length;
    }

    public double[] asArray(){
        return value;
    }

    private double[] value;
}
