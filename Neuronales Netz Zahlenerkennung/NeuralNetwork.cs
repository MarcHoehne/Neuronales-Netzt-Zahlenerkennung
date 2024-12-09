using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Neuronales_Netz_Zahlenerkennung
{
    public class NeuralNetwork
    {
        private List<Layer> layers = new List<Layer>();
        private const int InputSize = 14 * 14;
        private BatchHelper batchHelper;

        public NeuralNetwork(BatchHelper batchHelper)
        {
            this.batchHelper = batchHelper;
        }

        public void AddLayer(int inputSize, int outputSize)
        {
            layers.Add(new Layer(inputSize, outputSize));
        }

        public double[] Predict(double[] output)
        {
            foreach (var layer in layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        public void Train(byte[][,] images, byte[] labels, int epochs, double learningRate, int batchSize, int logFrequency)
        {
            int totalBatches = (int)Math.Ceiling((double)images.Length / batchSize);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
                //(byte[][,] shuffledImages, byte[] shuffledLabels) = ShuffleData(images, labels);

                double epochLoss = 0;
                for (int batch = 0; batch < totalBatches; batch++)
                {
                    int batchStart = batch * batchSize;
                    int batchEnd = Math.Min(batchStart + batchSize, images.Length);
                    var (batchImages, batchLabels) = batchHelper.GetBatch(images, labels, batchStart, batchEnd);

                    double batchLoss = batchHelper.ProcessBatch(batchImages, batchLabels, learningRate);
                    epochLoss += batchLoss;

                    if ((batch + 1) % logFrequency == 0)
                    {
                        Console.WriteLine($"  Batch {batch + 1}/{totalBatches} - Durchschnittlicher Loss: {batchLoss / batchSize:F4}");
                    }
                }

                Console.WriteLine($"Epoch {epoch + 1} - Gesamtdurchschnittlicher Loss: {epochLoss / images.Length:F4}");
            }
        }


        private (byte[][,] Images, byte[] Labels) ShuffleData(byte[][,] images, byte[] labels)
        {
            Random rng = new Random();
            int n = images.Length;
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (images[i], images[j]) = (images[j], images[i]);
                (labels[i], labels[j]) = (labels[j], labels[i]);
            }
            return (images, labels);
        }


        public double CrossEntropyLoss(double[] predicted, byte label)
        {
            // Überprüfen, ob der Index innerhalb des Label Bereichs ist
            if (label < 0 || label >= predicted.Length)
            {
                Console.WriteLine($"Warnung: Ungültiges Label {label}. Es sollte im Bereich von 0 bis {predicted.Length - 1} liegen.");
                return 0;  
            }
            // Epsilon für sehr kleine Werte um probleme mit Null zu verhindern!
            double epsilon = 1e-15;
            return -Math.Log(Math.Max(predicted[label], epsilon));
        }

        public void Backward(double[] input, byte label, double learningRate)
        {
            // Berechne die Ausgabe der Vorwärtspropagation
            double[] output = Predict(input);

            // Fehler für die Ausgangsschicht berechnen (Delta)
            if (label < 0 || label >= output.Length)
            {
                Console.WriteLine($"Warnung: Ungültiges Label {label} in Backward().");
                return;  
            }

            double[] delta = new double[output.Length];
            for (int j = 0; j < output.Length; j++)
            {
                delta[j] = output[j] - (j == label ? 1 : 0);  // Cross-Entropy-Loss für die Ausgangsschicht
            }

            // Gehe rückwärts durch alle Schichten
            for (int layerIndex = layers.Count - 1; layerIndex >= 0; layerIndex--)
            {
                var layer = layers[layerIndex];

                // Berechne den Fehler für die vorherige Schicht (Delta für die vorherige Schicht)
                double[] previousDelta = new double[layer.InputSize];

                for (int j = 0; j < layer.Biases.Length; j++)
                {
                    for (int i = 0; i < layer.InputSize; i++)
                    {
                        // Überprüfe, ob Indizes gültig sind
                        if (i >= layer.InputSize || j >= layer.Biases.Length)
                        {
                            Console.WriteLine($"Warnung: Ungültiger Index (i={i}, j={j}) in Backward().");
                            continue;  // Verhindere IndexOutOfRange
                        }

                        previousDelta[i] += delta[j] * layer.Weights[i, j];
                        // Update der Gewichte
                        if (i < layer.Weights.GetLength(0) && j < layer.Weights.GetLength(1))
                        {
                            layer.Weights[i, j] -= learningRate * delta[j] * (layerIndex == layers.Count - 1 ? input[i] : layers[layerIndex].Output[i]);
                        }
                    }
                    // Update der Biases
                    if (j < layer.Biases.Length)
                    {
                        layer.Biases[j] -= learningRate * delta[j];
                    }
                }

                // Weitergabe des Fehlerdelta an die vorherige Schicht
                delta = previousDelta;
            }
        }

    }
}