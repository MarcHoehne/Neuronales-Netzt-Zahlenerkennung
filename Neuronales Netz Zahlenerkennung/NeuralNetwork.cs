using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Neuronales_Netz_Zahlenerkennung
{
    public class NeuralNetwork
    {
        private readonly List<Layer> layers = new List<Layer>();
        private BatchHelper _batchHelper;

        public void Initialize(BatchHelper batchHelper)
        {
            _batchHelper = batchHelper ?? throw new ArgumentNullException(nameof(batchHelper));
        }

        public double CrossEntropyLoss(double[] predicted, int label)
        {
            const double epsilon = 1e-10; // Schutz vor Log(0)
            return -Math.Log(predicted[label] + epsilon);
        }

        public void Backpropagate(double[] input, double[] target, double learningRate)
        {
            double[] error = null;

            // Rückwärts durch die Layers
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                var layer = layers[i];

                if (i == layers.Count - 1) // Ausgabe-Layer
                {
                    // Fehler berechnen (target - output)
                    error = target.Zip(layer.Output, (t, o) => t - o).ToArray();
                }
                else
                {
                    // Fehler zurückpropagieren
                    error = layers[i + 1].Backward(error, layer.Weights);
                }

                // Gewichte aktualisieren
                layer.UpdateWeights(learningRate, error);
            }
        }


        public void AddLayer(int inputSize, int outputSize)
        {
            layers.Add(new Layer(inputSize, outputSize));
        }

        public double[] Predict(double[] input)
        {
            return layers.Aggregate(input, (current, layer) => layer.Forward(current));
        }

        public double Evaluate(byte[][,] images, byte[] labels)
        {
            int correct = 0;

            for (int i = 0; i < images.Length; i++)
            {
                var input = BatchHelper.NormalizeImage(images[i]);
                var output = Predict(input);
                int predicted = Array.IndexOf(output, output.Max());

                if (predicted == labels[i])
                    correct++;
            }

            return (double)correct / images.Length * 100;
        }

        public void SaveWeights(string filename)
        {
            using (StreamWriter writer = new StreamWriter(filename))
            {
                foreach (var layer in layers)
                {
                    for (int i = 0; i < layer.Weights.GetLength(0); i++)
                    {
                        for (int j = 0; j < layer.Weights.GetLength(1); j++)
                        {
                            writer.WriteLine(layer.Weights[i, j]);
                        }
                    }
                }
            }
        }

        public void LoadWeights(string filename)
        {
            if (!File.Exists(filename))
            {
                throw new FileNotFoundException($"Could not find weights file: {filename}");
            }

            string[] lines = File.ReadAllLines(filename);
            int lineIndex = 0;

            foreach (var layer in layers)
            {
                for (int i = 0; i < layer.Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < layer.Weights.GetLength(1); j++)
                    {
                        if (lineIndex >= lines.Length)
                        {
                            throw new Exception("Not enough weights in file");
                        }

                        layer.Weights[i, j] = double.Parse(lines[lineIndex]);
                        lineIndex++;
                    }
                }
            }
        }

        public void Train(byte[][,] images, byte[] labels, int epochs, double learningRate, int batchSize, int logFrequency)
        {
            int totalBatches = (int)Math.Ceiling((double)images.Length / batchSize);
            int imageCounter = 0; 

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Console.WriteLine($"Epoch {epoch}/{epochs}");
                double epochLoss = 0;

                Stopwatch stopwatch = Stopwatch.StartNew();

                int processorCount = Environment.ProcessorCount;
                int maxDegreeOfParallelism = Math.Max(1, processorCount - 2);
                ParallelOptions parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };


                for (int batch = 0; batch < totalBatches; batch++)
                {
                    int start = batch * batchSize;
                    int end = Math.Min(start + batchSize, images.Length);

                    var (batchImages, batchLabels) = _batchHelper.GetBatch(images, labels, start, end);

                    for (int i = 0; i < batchImages.Length; i++)
                    {
                        var input = BatchHelper.NormalizeImage(batchImages[i]);
                        var target = new double[10];
                        target[batchLabels[i]] = 1; // One-hot-Encoding für Labels

                        var output = Predict(input);
                        epochLoss += CrossEntropyLoss(output, batchLabels[i]);

                        Backpropagate(input, target, learningRate);

                        imageCounter++; // Increment the counter
                    }                
                }
                stopwatch.Stop(); // Stop the stopwatch

                Console.WriteLine($"Epoch {epoch} Complete - Avg Loss: {epochLoss / images.Length:F4} - Time: {stopwatch.Elapsed.TotalSeconds} seconds");
                Console.WriteLine($"Total images processed: {imageCounter}");
            }
        }

    }
}