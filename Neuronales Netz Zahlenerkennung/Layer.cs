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

        public double[] Input { get; private set; }
        public double[] Output { get; private set; }
        public double[,] Weights { get; private set; }
        public double[] Biases { get; private set; }
        public double[] PreActivation { get; private set; }
        public double[] Delta { get; private set; }

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

            // He initialization für bessere Gradienten
            double scale = Math.Sqrt(2.0 / inputSize);

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    Weights[i, j] = random.NextDouble() * 2 * scale - scale;
                }
            }

            for (int j = 0; j < outputSize; j++)
            {
                Biases[j] = 0;
            }
        }

        public double[] Forward(double[] input)
        {
            Input = input;
            Output = new double[outputSize];
            PreActivation = new double[outputSize];

            // Matrix multiplication für bessere Performance
            for (int j = 0; j < outputSize; j++)
            {
                double sum = Biases[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input[i] * Weights[i, j];
                }
                PreActivation[j] = sum;

                // ReLU für hidden layers, Softmax wird separat in der letzten Layer angewendet
                Output[j] = Math.Max(0, sum); // ReLU
            }

            return Output;
        }

        public void ComputeDelta(double[] error)
        {
            Delta = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
                Delta[i] = error[i] * Output[i] * (1 - Output[i]);
            }
        }

        public void ApplySoftmax()
        {
            double max = PreActivation.Max();
            double sum = 0;

            for (int i = 0; i < outputSize; i++)
            {
                Output[i] = Math.Exp(PreActivation[i] - max);
                sum += Output[i];
            }

            for (int i = 0; i < outputSize; i++)
            {
                Output[i] /= sum;
            }
        }

        public double[] BackwardOutput(double[] target)
        {
            Delta = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                // Softmax + Cross-Entropy Derivative
                Delta[i] = Output[i] - target[i];
            }
            return Delta;
        }

        public double[] BackwardHidden(double[] nextDelta, double[,] nextWeights)
        {
            Delta = new double[outputSize];

            for (int i = 0; i < outputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < nextDelta.Length; j++)
                {
                    sum += nextDelta[j] * nextWeights[i, j];
                }
                // ReLU derivative
                Delta[i] = sum * (PreActivation[i] > 0 ? 1 : 0);
            }

            return Delta;
        }

        public void UpdateWeights(double learningRate, double momentum = 0.9)
        {
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    double gradient = Delta[j] * Input[i];
                    Weights[i, j] -= learningRate * gradient;
                }
            }

            for (int j = 0; j < outputSize; j++)
            {
                Biases[j] -= learningRate * Delta[j];
            }
        }
    }
}