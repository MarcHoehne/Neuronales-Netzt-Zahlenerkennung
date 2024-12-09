using Neuronales_Netz_Zahlenerkennung;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class BatchHelper
{
    private const int InputSize = 14 * 14;
    private NeuralNetwork neuralNetwork;

    public BatchHelper(NeuralNetwork neuralNetwork)
    {
        this.neuralNetwork = neuralNetwork;
    }


    public (byte[][,] BatchImages, byte[] BatchLabels) GetBatch(byte[][,] images, byte[] labels, int start, int end)
    {
        int batchSize = end - start;
        var batchImages = new byte[batchSize][,];
        var batchLabels = new byte[batchSize];

        Array.Copy(images, start, batchImages, 0, batchSize);
        Array.Copy(labels, start, batchLabels, 0, batchSize);

        return (batchImages, batchLabels);
    }

    public double ProcessBatch(byte[][,] batchImages, byte[] batchLabels, double learningRate)
    {
        if (batchImages.Length != batchLabels.Length)
        {
            Console.WriteLine($"Fehler: Batchgröße von Bildern ({batchImages.Length}) stimmt nicht mit Batchgröße von Labels ({batchLabels.Length}) überein.");
            return -1;  
        }

        double batchLoss = 0;
        for (int i = 0; i < batchImages.Length; i++)
        {
            if (batchLabels.Length <= i)
            {
                Console.WriteLine("Warnung: Batch-Label ist ungültig (Index außerhalb des Bereichs).");
                continue;  
            }

            double[] input = NormalizeImage(batchImages[i]);
            // Sicherstellen, dass das Bild korrekt flach gemacht wurde
            if (input == null || input.Length != InputSize)
            {
                Console.WriteLine($"Warnung: Eingabebild {i} ist ungültig (flach gemacht).");
                continue;  
            }
            //Berechnet den Batchloss aus dem Vorwärtsschritt durch alle labels
            double[] predicted = neuralNetwork.Predict(input);
            batchLoss += neuralNetwork.CrossEntropyLoss(predicted, batchLabels[i]);

            // Fehlerbehandlung für die Backpropagation
            try
            {
                neuralNetwork.Backward(input, batchLabels[i], learningRate);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Fehler in Backward() für Index {i}: {ex.Message}");
                continue;  
            }
        }

        return batchLoss;
    }

    /// <summary>
    /// Normalisiert das Eingabebild, indem jeder Pixelwert durch 255 geteilt wird, um Werte im Bereich [0, 1] zu erhalten. 
    /// Das Bild muss die Größe 14x14 haben. 
    /// </summary>
    /// <param name="image">Das Eingabebild als 2D-Array (14x14).</param>
    /// <returns>Ein Array von normalisierten Werten, das die Pixel des Bildes repräsentiert.</returns>
    public static double[] NormalizeImage(byte[,] image)
    {
        int width = image.GetLength(1);
        int height = image.GetLength(0);

        // Sicherstellen, dass das Bild 14x14 ist
        if (width != 14 || height != 14)
        {
            Console.WriteLine($"Warnung: Bildgröße ist falsch. Erwartet 14x14, aber erhalten {width}x{height}");
            return null;  // Bild ist ungültig
        }

        double[] normalized = new double[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                normalized[y * width + x] = image[y, x] / 255.0;  // Normalisierung: Werte zwischen 0 und 1
            }
        }

        return normalized;
    }


}

