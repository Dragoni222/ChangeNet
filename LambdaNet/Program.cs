//There are three questions: How does this change the number,
//how does everything else change this, and what do I consider next.

//The second question is very hard because it's infinitely recursive- you have 
//to ask the question again all the way down.

//Maybe you start by considering all the variables' relations first, in an arbitrary order,
//then every time you change something you start over and consider again.

//So give every possible change to a value a weight for each other variable
//then use gradient decent



using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;
using Vector = MathNet.Numerics.LinearAlgebra.Double.Vector;


Node[] network = new Node[3];
for (int i = 0; i < 3; i++)
{
    network[i] = new Node(3);
}

Random random = new Random();
for (int i = 0; i < 10000; i++)
{
    int startValue = random.Next(0,10);
    double[] expected = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    expected[startValue] = 1;
    double[] outputs = network[0].ChangeConfidencesTrain(new double[] { startValue, 2, 3}, expected, 1);
    Console.WriteLine("\n\n Expected: "+ outputs[startValue] +" Full:" + string.Join(", ", outputs));
}


    

class Node
{
    private Matrix<double> Weights;  //one weight vector for each number 
    private Vector<double> Biases; //One vector for the biases

    public Node(int totalNodes)
    {
        Weights = Matrix<double>.Build.Random(10, totalNodes);
        Biases = Vector<double>.Build.Random(10);

    }

    public double[] ChangeConfidences(double[] inputValues)
    {
        Vector<double> values = Vector<double>.Build.DenseOfArray(inputValues); // totalNodes length start vector

        Vector<double> resultant = values * Weights;
        Vector<double> withBiases = resultant + Biases;
        return withBiases.Map(result => Sigmoid(result)).ToArray();
    }

    public double[] ChangeConfidencesTrain(double[] inputValues, double[] expectedValues, double trainSpeed)
    {
        Vector<double> values = Vector<double>.Build.DenseOfArray(inputValues); // Input vector
        Vector<double> resultant = Weights * values; // Weighted sum
        Vector<double> zValues = resultant + Biases; // Adding biases
        Vector<double> final = zValues.Map(result => Sigmoid(result)); // Applying activation function
        for (int i = 0; i < Biases.Count; i++)
        {
            double loss = final[i] - expectedValues[i];
            double sigmoidDerivative = final[i] * (1 - final[i]);
            Biases[i] -= loss * sigmoidDerivative * trainSpeed;
            for (int j = 0; j < Weights.ColumnCount; j++)
            {
                Weights[i, j] -= loss * values[j] * sigmoidDerivative * trainSpeed;
            }
            
        }

        return final.ToArray();
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

