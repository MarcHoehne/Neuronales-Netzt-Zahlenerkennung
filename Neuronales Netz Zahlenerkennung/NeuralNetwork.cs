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
            // Forward pass speichern
            var layerOutputs = new List<double[]>();
            var currentInput = input;

            foreach (var layer in layers)
            {
                currentInput = layer.Forward(currentInput);
                layerOutputs.Add(currentInput);
            }

            // Backpropagation
            double[] error = null;

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                if (i == layers.Count - 1)
                {
                    error = new double[layers[i].Output.Length];
                    for (int j = 0; j < error.Length; j++)
                    {
                        error[j] = layers[i].Output[j] - target[j];
                    }
                }
                else
                {
                    // Hidden Layers: Propagiere Error zurück
                    var nextLayer = layers[i + 1];
                    error = new double[layers[i].Output.Length];

                    for (int j = 0; j < error.Length; j++)
                    {
                        double sum = 0;
                        for (int k = 0; k < nextLayer.Output.Length; k++)
                        {
                            sum += nextLayer.Weights[j, k] * nextLayer.Delta[k];
                        }
                        error[j] = sum;
                    }
                }
                // Berechne Delta für aktuelle Layer
                layers[i].ComputeDelta(error);

                // Update Weights und Biases - Fixed method call
                layers[i].UpdateWeights(learningRate);
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
                    // Speichern der Gewichtsmatrix
                    for (int i = 0; i < layer.Weights.GetLength(0); i++)
                    {
                        for (int j = 0; j < layer.Weights.GetLength(1); j++)
                        {
                            writer.WriteLine(layer.Weights[i, j]);
                        }
                    }

                    // Speichern der Bias-Werte
                    for (int j = 0; j < layer.Biases.Length; j++)
                    {
                        writer.WriteLine(layer.Biases[j]);
                    }
                }
            }

            Console.WriteLine("Weights saved to file.");
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
                // Laden der Gewichtsmatrix
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

                // Laden der Bias-Werte
                for (int j = 0; j < layer.Biases.Length; j++)
                {
                    if (lineIndex >= lines.Length)
                    {
                        throw new Exception("Not enough bias values in file");
                    }

                    layer.Biases[j] = double.Parse(lines[lineIndex]);
                    lineIndex++;
                }
            }

            Console.WriteLine("Weights loaded from file.");
        }

        public void CompareWeights(int layerIndex, string weightsFile)
        {
            // Originalgewichte abrufen
            double[,] originalWeights = GetLayerWeights(layerIndex);

            // Gewichte laden
            LoadWeights(weightsFile);

            // Geladene Gewichte abrufen
            double[,] loadedWeights = GetLayerWeights(layerIndex);

            // Vergleich der Gewichte
            for (int i = 0; i < originalWeights.GetLength(0); i++)
            {
                for (int j = 0; j < originalWeights.GetLength(1); j++)
                {
                    if (originalWeights[i, j] != loadedWeights[i, j])
                    {
                        Console.WriteLine("Mismatch at weight position: ({0}, {1})", i, j);
                    }
                }
            }
        }

        public double[,] GetLayerWeights(int layerIndex)
        {
            if (layerIndex < 0 || layerIndex >= layers.Count)
            {
                throw new ArgumentOutOfRangeException($"Layer index {layerIndex} is out of range.");
            }

            return layers[layerIndex].Weights;
        }

        public (double Loss, double Accuracy) EvaluateBatch(byte[][,] images, byte[] labels)
        {
            int correct = 0;
            double totalLoss = 0;

            for (int i = 0; i < images.Length; i++)
            {
                var input = BatchHelper.NormalizeImage(images[i]);
                var output = Predict(input);
                int predicted = Array.IndexOf(output, output.Max());

                if (predicted == labels[i])
                    correct++;

                totalLoss += CrossEntropyLoss(output, labels[i]);
            }

            double accuracy = (double)correct / images.Length * 100;
            double averageLoss = totalLoss / images.Length;

            return (averageLoss, accuracy);
        }

        public void Train(byte[][,] images, byte[] labels, int epochs, double learningRate, int batchSize, int logFrequency)
        {
            int totalBatches = (int)Math.Ceiling((double)images.Length / batchSize);

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Console.WriteLine($"\nEpoch {epoch}/{epochs}");
                double epochLoss = 0;
                int epochCorrect = 0;

                // Learning rate decay
                double currentLearningRate = learningRate * (1.0 / (1.0 + 0.0001 * epoch));

                // Shuffle training data
                var indices = Enumerable.Range(0, images.Length).ToArray();
                Random rng = new Random();
                indices = indices.OrderBy(x => rng.Next()).ToArray();

                Stopwatch stopwatch = Stopwatch.StartNew();

                for (int batch = 0; batch < totalBatches; batch++)
                {
                    int start = batch * batchSize;
                    int end = Math.Min(start + batchSize, images.Length);
                    int currentBatchSize = end - start;

                    double batchLoss = 0;
                    int batchCorrect = 0;

                    // Process batch
                    for (int i = start; i < end; i++)
                    {
                        var input = BatchHelper.NormalizeImage(images[indices[i]]);
                        var target = new double[10];
                        target[labels[indices[i]]] = 1;

                        // Forward pass
                        var current = input;
                        for (int l = 0; l < layers.Count; l++)
                        {
                            current = layers[l].Forward(current);
                            if (l == layers.Count - 1)
                            {
                                layers[l].ApplySoftmax();
                                current = layers[l].Output;
                            }
                        }

                        // Calculate accuracy and loss
                        int predicted = Array.IndexOf(current, current.Max());
                        if (predicted == labels[indices[i]])
                            batchCorrect++;

                        // Cross-entropy loss
                        batchLoss -= Math.Log(current[labels[indices[i]]] + 1e-10);

                        // Backward pass
                        var error = layers[layers.Count - 1].BackwardOutput(target);
                        for (int l = layers.Count - 2; l >= 0; l--)
                        {
                            error = layers[l].BackwardHidden(error, layers[l + 1].Weights);
                        }

                        // Update weights
                        for (int l = 0; l < layers.Count; l++)
                        {
                            layers[l].UpdateWeights(currentLearningRate);
                        }
                    }

                    epochLoss += batchLoss;
                    epochCorrect += batchCorrect;

                    if ((batch + 1) % logFrequency == 0)
                    {
                        double batchAccuracy = (double)batchCorrect / currentBatchSize * 100;
                        Console.WriteLine($"Batch {batch + 1}/{totalBatches} - Loss: {batchLoss / currentBatchSize:F4}, Accuracy: {batchAccuracy:F2}%");
                    }
                }

                stopwatch.Stop();

                Console.WriteLine($"\nEpoch {epoch} Summary:");
                Console.WriteLine($"Average Loss: {epochLoss / images.Length:F4}");
                Console.WriteLine($"Training Accuracy: {(double)epochCorrect / images.Length * 100:F2}%");
                Console.WriteLine($"Time: {stopwatch.Elapsed.TotalSeconds:F1} seconds");
            }
        }
    }
}