using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neuronales_Netz_Zahlenerkennung
{
    public class MNISTImageViewer : Form
    {
        private TextBox imageIndexTextBox;
        private Button showImageButton;
        private PictureBox imagePictureBox;
        private Label actualLabelLabel;
        private Label predictedLabelLabel;
        private NeuralNetwork neuralNetwork;
        private Label modelAccuracyLabel;
        private (byte[][,] Images, byte[] Labels) testData;

        public MNISTImageViewer(NeuralNetwork network, (byte[][,] Images, byte[] Labels) testData)
        {
            this.neuralNetwork = network;
            this.testData = testData;
            InitializeComponents();
        }

        private void InitializeComponents()
        {
            this.Size = new Size(400, 500);
            this.Text = "MNIST Image Viewer";

            // Image Index Input
            Label indexLabel = new Label
            {
                Text = "Enter Image Index (0-60000):",
                Location = new Point(20, 20),
                Size = new Size(150, 20)
            };
            this.Controls.Add(indexLabel);

            imageIndexTextBox = new TextBox
            {
                Location = new Point(180, 20),
                Size = new Size(100, 20)
            };
            this.Controls.Add(imageIndexTextBox);

            // Show Image Button
            showImageButton = new Button
            {
                Text = "Show Image",
                Location = new Point(290, 20),
                Size = new Size(80, 23)
            };
            showImageButton.Click += ShowImageButton_Click;
            this.Controls.Add(showImageButton);

            // PictureBox for displaying the image
            imagePictureBox = new PictureBox
            {
                Location = new Point(50, 60),
                Size = new Size(280, 280),
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.Zoom
            };
            this.Controls.Add(imagePictureBox);

            // Labels for actual and predicted values
            actualLabelLabel = new Label
            {
                Location = new Point(50, 360),
                Size = new Size(280, 20),
                Text = "Actual Label: ",
                Font = new Font(FontFamily.GenericSansSerif, 12f)
            };
            this.Controls.Add(actualLabelLabel);

            predictedLabelLabel = new Label
            {
                Location = new Point(50, 390),
                Size = new Size(280, 20),
                Text = "Network Prediction: ",
                Font = new Font(FontFamily.GenericSansSerif, 12f)
            };
            this.Controls.Add(predictedLabelLabel);

            // Füge Label für Modellgenauigkeit hinzu
            modelAccuracyLabel = new Label
            {
                Location = new Point(50, 420),
                Size = new Size(280, 20),
                Text = "Model Accuracy: ",
                Font = new Font(FontFamily.GenericSansSerif, 12f)
            };
            this.Controls.Add(modelAccuracyLabel);

            // Berechne und zeige die Gesamtgenauigkeit beim Start
            double accuracy = neuralNetwork.Evaluate(testData.Images, testData.Labels);
            modelAccuracyLabel.Text = $"Model Accuracy: {accuracy:F2}%";
        }

        private void ShowImageButton_Click(object sender, EventArgs e)
        {
            if (!int.TryParse(imageIndexTextBox.Text, out int index) ||
                index < 0 ||
                index >= testData.Images.Length)
            {
                MessageBox.Show($"Please enter a valid index between 0 and {testData.Images.Length - 1}");
                return;
            }

            // Get the image and create a bitmap
            byte[,] mnistImage = testData.Images[index];
            Bitmap bmp = new Bitmap(28, 28);

            // Convert MNIST data to bitmap
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    int pixelValue = mnistImage[y / 2, x / 2]; // Since we're using reduced resolution
                    Color pixelColor = Color.FromArgb(pixelValue, pixelValue, pixelValue);
                    bmp = new Bitmap(bmp, new Size(280, 280)); // Scale up for better visibility
                    bmp.SetPixel(x, y, pixelColor);
                }
            }

            imagePictureBox.Image = bmp;

            // Get actual label
            int actualLabel = testData.Labels[index];
            actualLabelLabel.Text = $"Actual Label: {actualLabel}";

            // Get network prediction
            var input = BatchHelper.NormalizeImage(mnistImage);
            var output = neuralNetwork.Predict(input);
            int predictedLabel = Array.IndexOf(output, output.Max());
            double confidence = output[predictedLabel] * 100;

            predictedLabelLabel.Text = $"Network Prediction: {predictedLabel} (Confidence: {confidence:F1}%)";
        }
    }
}
