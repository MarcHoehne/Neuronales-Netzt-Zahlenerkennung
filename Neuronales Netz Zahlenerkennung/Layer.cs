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
        private readonly RechenMethoden rechenmethoden = new RechenMethoden();

        private int inputSize;
        private int outputSize;

        public double[] Input { get; private set; } // Speichert die Eingabe der Forward-Methode
        public double[] Output { get; private set; } // Ausgabe nach Forward-Pass
        public double[,] Weights { get; private set; } // Gewichtsmatrix
        public double[] Biases { get; private set; } // Biases

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
            if (input.Length != inputSize)
                throw new ArgumentException($"Expected input size: {inputSize}, got: {input.Length}");

            Input = input; // Speichert die Eingabe für Backpropagation
            Output = new double[outputSize];

            for (int j = 0; j < outputSize; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input[i] * Weights[i, j];
                }
                sum += Biases[j];
                Output[j] = rechenmethoden.Sigmoid(sum); // Aktivierungsfunktion
            }

            return Output;
        }

        public double[] Backward(double[] error, double[,] weights)
        {
            // Fehler für die vorherige Schicht berechnen
            double[] prevError = new double[inputSize];
            double[,] gradients = new double[inputSize, outputSize];

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    gradients[i, j] = error[j] * Output[j] * (1 - Output[j]); // Ableitung der Sigmoid-Funktion
                    prevError[i] += gradients[i, j] * weights[i, j]; // Fehler für die Eingabe
                }
            }

            return prevError;
        }

        public void UpdateWeights(double learningRate, double[] error)
        {
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    double gradient = error[j] * Output[j] * (1 - Output[j]); // Gradientenberechnung
                    Weights[i, j] += learningRate * gradient * Input[i]; // Gewichte aktualisieren
                }
            }

            for (int j = 0; j < outputSize; j++)
            {
                Biases[j] += learningRate * error[j]; // Biases aktualisieren
            }
        }
    }
}