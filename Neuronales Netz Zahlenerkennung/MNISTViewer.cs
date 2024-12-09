using static System.Net.Mime.MediaTypeNames;
using System.Drawing;
using System.Windows.Forms;
using System;
using Application = System.Windows.Forms.Application;
using Neuronales_Netz_Zahlenerkennung;
using System.Linq;
public class MNISTViewer
{
    private static (byte[][,] Images, byte[] Labels) mnistData;

    [STAThread]
    public static void Main()
    {
        Console.WriteLine("Initialisiere Netzwerk...");

        BatchHelper batchHelper = new BatchHelper(null);

        NeuralNetwork neuralNetwork = new NeuralNetwork(batchHelper);

        batchHelper = new BatchHelper(neuralNetwork);

        // Lade MNIST-Daten und bereite sie vor
        var mnistData = MNISTReader.ReadTrainingData();
        var testData = MNISTReader.ReadTestData();

        const int InputSize = 14 * 14; // wird schon in ReadTrainingData und ReadTestData zur hälfte reduziert!
        const int OutputSize = 10; // Anzahl der Klassen (0-9 für MNIST)

        neuralNetwork.AddLayer(InputSize,10); // Eingabe zu erstem Layer
        //neuralNetwork.AddLayer(512, 256);
        //neuralNetwork.AddLayer(256, 128);
        //neuralNetwork.AddLayer(128, 64);
        //neuralNetwork.AddLayer(64, OutputSize); // Letzter Layer zu Ausgabe (10 Klassen)

        // Trainiere das Netzwerk
        neuralNetwork.Train(
            mnistData.Images, // Bilder 
            mnistData.Labels, // Labels für die Bilder
            epochs: 5,       // Anzahl der Epochen
            learningRate: 0.01, // Lernrate
            batchSize: 64,    // Batchgröße
            logFrequency: 50  // Log-Häufigkeit
        );

        Console.WriteLine("Bewerte Modell...");
        Evaluate(neuralNetwork, testData.Images, testData.Labels);

        Console.WriteLine("Modell fertig trainiert.");
    }

    private static void Evaluate(NeuralNetwork neuralNetwork, byte[][,] testImages, byte[] testLabels)
    {
        int correct = 0;
        for (int i = 0; i < testImages.Length; i++)
        {
            double[] input = MNISTViewer.FlattenImage(testImages[i]);
            double[] output = neuralNetwork.Predict(input);
            int predicted = Array.IndexOf(output, output.Max());

            if (predicted == testLabels[i])
            {
                correct++;
            }
        }

        double accuracy = (double)correct / testImages.Length * 100;
        Console.WriteLine($"Genauigkeit: {accuracy:F2}%");
    }

    // Initialisiert die Gewichte zufällig
    private static double[,] InitializeWeights(int inputSize, int outputSize)
    {
        Random random = new Random();
        double[,] weights = new double[inputSize, outputSize];

        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                weights[i, j] = random.NextDouble() * 2 - 1;  // Werte zwischen -1 und 1
            }
        }

        return weights;
    }

    // Initialisiert die Biases zufällig
    private static double[] InitializeBiases(int outputSize)
    {
        Random random = new Random();
        double[] biases = new double[outputSize];

        for (int j = 0; j < outputSize; j++)
        {
            biases[j] = random.NextDouble() * 2 - 1;
        }

        return biases;
    }

    public static double[] FlattenImage(byte[,] image)
    {
        int width = image.GetLength(1);
        int height = image.GetLength(0);

        if (width != 14 || height != 14)
        {
            throw new ArgumentException($"Image size is incorrect. Expected 14x14 but got {width}x{height}");
        }

        double[] flattened = new double[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                flattened[y * width + x] = image[y, x] / 255.0;  // Normalisierung auf Werte zwischen 0 und 1
            }
        }

        return flattened;
    }

}