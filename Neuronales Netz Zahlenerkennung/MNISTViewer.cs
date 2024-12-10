using static System.Net.Mime.MediaTypeNames;
using System.Drawing;
using System.Windows.Forms;
using System;
using Application = System.Windows.Forms.Application;
using Neuronales_Netz_Zahlenerkennung;
using System.Linq;
using System.IO;

public class MNISTViewer
{
    private static (byte[][,] Images, byte[] Labels) mnistData;

    [STAThread]    
    static void Main()
    {
        Console.WriteLine("Initializing Neural Network...");

        // Create Neural Network and Helper
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        BatchHelper batchHelper = new BatchHelper();

        neuralNetwork.Initialize(batchHelper);
        batchHelper.Initialize(neuralNetwork);

        // Load MNIST Data
        Console.WriteLine("Loading MNIST Data...");
        var trainingData = MNISTReader.ReadTrainingData();
        var testData = MNISTReader.ReadTestData();

        // Print the number of training and test images
        Console.WriteLine($"Number of training images: {trainingData.Images.Length}");
        Console.WriteLine($"Number of test images: {testData.Images.Length}");


        // Configure Network
        const int inputSize = 14 * 14; // Reduced image size
        const int outputSize = 10; // Number of classes (0-9)

        neuralNetwork.AddLayer(inputSize, 7*7);
        neuralNetwork.AddLayer(7*7, outputSize);

        const string weightsFile = "Weights2.txt";

        if (File.Exists(weightsFile))
        {
            Console.WriteLine("Loading weights from file...");
            neuralNetwork.LoadWeights(weightsFile);
        }
        else
        {
            Console.WriteLine("Starting Training...");
            neuralNetwork.Train(
                trainingData.Images,
                trainingData.Labels,
                epochs: 70,
                learningRate: 0.01,
                batchSize: 64,
                logFrequency: 50
            );

            Console.WriteLine("Saving weights to file...");
            neuralNetwork.SaveWeights(weightsFile);
        }

        // Evaluate the trained model
        Console.WriteLine("Evaluating Model...");
        double accuracy = neuralNetwork.Evaluate(testData.Images, testData.Labels);
        Console.WriteLine($"Final Model Accuracy: {accuracy:F2}%");
        Console.ReadLine();
    }
}

