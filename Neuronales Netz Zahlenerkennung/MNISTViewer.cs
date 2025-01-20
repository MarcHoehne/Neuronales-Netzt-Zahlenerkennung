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

        neuralNetwork.Initialize(batchHelper);
        batchHelper.Initialize(neuralNetwork);

        // Load MNIST Data
        Console.WriteLine("Loading MNIST Data...");
        var trainingData = MNISTReader.ReadTrainingData();
        var testData = MNISTReader.ReadTestData();

        neuralNetwork.AddLayer(inputSize, 128);    // Erste Hidden Layer
        neuralNetwork.AddLayer(128, 64);           // Zweite Hidden Layer
        neuralNetwork.AddLayer(64, outputSize);     // Output Layer

        // Load the weights if they exist
        string weightsFile = "trained_weights20.01.1425.txt"; // Use a fixed name for consistency
        if (File.Exists(weightsFile))
        {
            Console.WriteLine("Loading pre-trained weights...");
            neuralNetwork.LoadWeights(weightsFile);
            // In Main()
            double accuracy = neuralNetwork.Evaluate(testData.Images, testData.Labels);
            Console.WriteLine($"Final Model Accuracy: {accuracy:F2}%");
        }
        else
        {
            Console.WriteLine("No pre-trained weights found. Please train the network first.");
            HandleWeightsAndTraining(neuralNetwork, trainingData, testData, weightsFile);
            neuralNetwork.LoadWeights(weightsFile);
            // In Main()
            double accuracy = neuralNetwork.Evaluate(testData.Images, testData.Labels);
            Console.WriteLine($"Final Model Accuracy: {accuracy:F2}%");
        }

        // Create and show the image viewer form
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new MNISTImageViewer(neuralNetwork, testData));
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
                epochs: 10,
                learningRate: 0.01,
                batchSize: 64,
                logFrequency: 50
            );

            Console.WriteLine("Saving weights to file...");
            neuralNetwork.SaveWeights(weightsFile);
            Console.ReadLine();
        }
    }
}