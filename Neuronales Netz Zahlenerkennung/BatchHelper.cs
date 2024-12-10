using Neuronales_Netz_Zahlenerkennung;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public class BatchHelper
{
    private NeuralNetwork _neuralNetwork;

    public void Initialize(NeuralNetwork neuralNetwork)
    {
        _neuralNetwork = neuralNetwork ?? throw new ArgumentNullException(nameof(neuralNetwork));
    }

    public (byte[][,] Images, byte[] Labels) GetBatch(byte[][,] images, byte[] labels, int start, int end)
    {
        int batchSize = end - start;
        byte[][,] batchImages = new byte[batchSize][,];
        byte[] batchLabels = new byte[batchSize];

        Array.Copy(images, start, batchImages, 0, batchSize);
        Array.Copy(labels, start, batchLabels, 0, batchSize);

        return (batchImages, batchLabels);
    }

    public static double[] NormalizeImage(byte[,] image)
    {
        int width = image.GetLength(1);
        int height = image.GetLength(0);

        if (width != 14 || height != 14)
            throw new ArgumentException($"Image size incorrect. Expected 14x14, got {width}x{height}");

        double[] normalized = new double[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                normalized[y * width + x] = image[y, x] / 255.0;

        return normalized;
    }

}
