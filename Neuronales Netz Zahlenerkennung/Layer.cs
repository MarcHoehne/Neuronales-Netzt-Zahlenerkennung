using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RechenmethodenMatrix;

namespace Neuronales_Netz_Zahlenerkennung
{
    public class Layer
    {
        RechenMethoden rechenmethoden = new RechenMethoden();

        // Felder für InputSize und OutputSize
        private int inputSize;
        private int outputSize;

        public int InputSize => inputSize; // Getter für inputSize
        public double[] Output { get; private set; } // Getter und Setter für Output

        public double[,] Weights { get; private set; }
        public double[] Biases { get; private set; }

        public Layer(int inputSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;

            InitializeWeightsAndBiases();
        }

        private void InitializeWeightsAndBiases()
        {
            Random random = new Random();
            Weights = new double[inputSize, outputSize];
            Biases = new double[outputSize];

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    Weights[i, j] = random.NextDouble() * Math.Sqrt(2.0 / inputSize); // He-Initialisierung
                }
            }

            for (int j = 0; j < outputSize; j++)
            {
                Biases[j] = random.NextDouble() * 2 - 1;
            }
        }

        public double[] Forward(double[] input)
        {
            // Dimensionscheck
            if (input.Length != inputSize)
            {
                throw new ArgumentException($"Ungültige Eingabedimension. Erwartet: {inputSize}, erhalten: {input.Length}");
            }

            double[] output = new double[outputSize];

            for (int j = 0; j < outputSize; j++)
            {
                double sum = 0.0;

                for (int i = 0; i < inputSize; i++)
                {
                    sum += input[i] * Weights[i, j];
                }

                sum += Biases[j];
                output[j] = rechenmethoden.Sigmoid(sum); // Sigmoid-Aktivierungsfunktion
            }

            this.Output = output; // Ausgabe speichern
            return output;
        }
    }
}