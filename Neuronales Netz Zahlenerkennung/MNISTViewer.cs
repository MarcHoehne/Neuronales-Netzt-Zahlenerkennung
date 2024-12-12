using static System.Net.Mime.MediaTypeNames;
using System.Drawing;
using System.Windows.Forms;
using System;
using Application = System.Windows.Forms.Application;
using Neuronales_Netz_Zahlenerkennung;
using System.Linq;
using System.IO;
using static System.Windows.Forms.LinkLabel;
using System.Reflection.Emit;

public class MNISTViewer
{
    private static (byte[][,] Images, byte[] Labels) mnistData;

    [STAThread]
    static void Main()
    {
        Console.WriteLine("Initializing Neural Network...");

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        BatchHelper batchHelper = new BatchHelper();

        const int inputSize = 14 * 14;
        const int outputSize = 10;

        Layer layer = new Layer(inputSize, outputSize);

        neuralNetwork.Initialize(batchHelper);
        batchHelper.Initialize(neuralNetwork);

        // Load MNIST Data
        Console.WriteLine("Loading MNIST Data...");
        var trainingData = MNISTReader.ReadTrainingData();
        var testData = MNISTReader.ReadTestData();

        Console.WriteLine($"Number of training images: {trainingData.Images.Length}");
        Console.WriteLine($"Number of test images: {testData.Images.Length}");


        neuralNetwork.AddLayer(inputSize, 7 * 7);
        neuralNetwork.AddLayer(7 * 7, outputSize);

        const string weightsFile = "Weights.txt";

        HandleWeightsAndTraining(neuralNetwork, trainingData, testData, weightsFile);

    }

    private static void HandleWeightsAndTraining(NeuralNetwork neuralNetwork, (byte[][,] Images, byte[] Labels) trainingData, (byte[][,] Images, byte[] Labels) testData, string weightsFile)
    {
        if (File.Exists(weightsFile))
        {
            Console.WriteLine("Weights file found. Do you want to continue training with the existing weights? (Y/N)");
            string input = Console.ReadLine();

            if (input?.ToUpper() == "Y")
            {
                Console.WriteLine("Loading weights from file...");
                neuralNetwork.LoadWeights(weightsFile);

                Console.WriteLine("Comparing saved and loaded weights...");
                neuralNetwork.CompareWeights(0, weightsFile);
                double oldAccuracy = neuralNetwork.Evaluate(testData.Images, testData.Labels);
                Console.WriteLine($"Actuall Model Accuracy: {oldAccuracy:F2}%");


                Console.WriteLine("Continuing training with the saved weights...");
                neuralNetwork.Train(
                    trainingData.Images,
                    trainingData.Labels,
                    epochs: 10,
                    learningRate: 0.01,
                    batchSize: 64,
                    logFrequency: 50
                );

                Console.WriteLine("Saving updated weights...");
                neuralNetwork.SaveWeights(weightsFile);
                double accuracy = neuralNetwork.Evaluate(testData.Images, testData.Labels);
                Console.WriteLine($"Model Accuracy: {accuracy:F2}%");
                Console.ReadLine();
            }
            else if (input?.ToUpper() == "N")
            {
                Console.WriteLine("Evaluating the model with the loaded weights...");
                neuralNetwork.LoadWeights(weightsFile); 

                double accuracy = neuralNetwork.Evaluate(testData.Images, testData.Labels);
                Console.WriteLine($"Model Accuracy: {accuracy:F2}%");
                Console.ReadLine();
            }
            else
            {
                Console.WriteLine("Invalid input.");
                Console.ReadLine();
            }
        }
        else
        {
            Console.WriteLine("No weights found. Starting training from scratch...");

            neuralNetwork.Train(
                trainingData.Images,
                trainingData.Labels,
                epochs: 6,
                learningRate: 0.01,
                batchSize: 64,
                logFrequency: 50
            );

            Console.WriteLine("Saving weights to file...");
            neuralNetwork.SaveWeights(weightsFile);
        }
    }
}