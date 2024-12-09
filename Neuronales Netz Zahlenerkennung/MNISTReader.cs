using System.IO;
using System;

public static class MNISTReader
{
    private const string TrainImagesPath = "mnist/train-images.idx3-ubyte";
    private const string TrainLabelsPath = "mnist/train-labels.idx1-ubyte";
    private const string TestImagesPath = "mnist/t10k-images.idx3-ubyte";
    private const string TestLabelsPath = "mnist/t10k-labels.idx1-ubyte";

    public static (byte[][,] Images, byte[] Labels) ReadTrainingData()
    {
        return ReadData(TrainImagesPath, TrainLabelsPath);
    }

    public static (byte[][,] Images, byte[] Labels) ReadTestData()
    {
        return ReadData(TestImagesPath, TestLabelsPath);
    }

    private static (byte[][,] Images, byte[] Labels) ReadData(string imagesPath, string labelsPath)
    {
        using (BinaryReader labelsReader = new BinaryReader(new FileStream(labelsPath, FileMode.Open)))
        using (BinaryReader imagesReader = new BinaryReader(new FileStream(imagesPath, FileMode.Open)))
        {
            // Skip the headers
            imagesReader.ReadBigInt32(); // Magic number for images
            int numberOfImages = imagesReader.ReadBigInt32();
            int imageWidth = imagesReader.ReadBigInt32();
            int imageHeight = imagesReader.ReadBigInt32();

            labelsReader.ReadBigInt32(); // Magic number for labels
            int numberOfLabels = labelsReader.ReadBigInt32();

            byte[][,] images = new byte[numberOfImages][,];
            byte[] labels = new byte[numberOfImages];

            for (int i = 0; i < numberOfImages; i++)
            {
                byte[,] image = new byte[imageHeight, imageWidth];

                // Read image pixel data
                for (int row = 0; row < imageHeight; row++)
                {
                    for (int col = 0; col < imageWidth; col++)
                    {
                        image[row, col] = imagesReader.ReadByte();
                    }
                }

                // Reduce the image resolution by half and store it
                images[i] = ReduceResolution(image);
                labels[i] = labelsReader.ReadByte();
            }

            return (images, labels);
        }
    }

    // Reduce image resolution by averaging every 2x2 block
    public static byte[,] ReduceResolution(byte[,] original)
    {
        int originalWidth = original.GetLength(1);
        int originalHeight = original.GetLength(0);
        byte[,] reduced = new byte[originalHeight / 2, originalWidth / 2];

        for (int i = 0; i < originalHeight; i += 2)
        {
            for (int j = 0; j < originalWidth; j += 2)
            {
                int sum = original[i, j] + original[i + 1, j] + original[i, j + 1] + original[i + 1, j + 1];
                reduced[i / 2, j / 2] = (byte)(sum / 4);
            }
        }

        return reduced;
    }

    // Method to handle big-endian integers in the MNIST files
    private static int ReadBigInt32(this BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(bytes);
        }
        return BitConverter.ToInt32(bytes, 0);
    }
}
