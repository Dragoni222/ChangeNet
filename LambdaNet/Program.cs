//There are three questions: How does this change the number,
//how does everything else change this, and what do I consider next.

//The second question is very hard because it's infinitely recursive- you have 
//to ask the question again all the way down.

//Maybe you start by considering all the variables' relations first, in an arbitrary order,
//then every time you change something you start over and consider again.

//So give every possible change to a value a weight for each other variable
//then use gradient decent



using MathNet.Numerics.LinearAlgebra;
class Node
{
    private Matrix<double> Weights;  //one weight vector for each number 
    private Vector<double> Biases; //One vector for the biases

    public Node(int totalNodes)
    {
        Weights = Matrix<double>.Build.Dense(totalNodes, 10); //Maybe reversed?
        Biases = Vector<double>.Build.Dense(totalNodes);

    }

    public double[] ChangeConfidences(double[] currentValues)
    {
        Vector<double> values = Vector<double>.Build.DenseOfArray(currentValues); // totalNodes length start vector

        Vector<double> resultant = values * Weights;
        Vector<double> withBiases = resultant + Biases;
        return withBiases.Map(result => Sigmoid(result)).ToArray();
    }
    
    static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    static double SigmoidDerivative(double sigmoidValue)
    {
        return sigmoidValue * (1 - sigmoidValue);
    }

}