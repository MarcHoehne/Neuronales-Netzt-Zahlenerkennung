using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuronales_Netz_Zahlenerkennung
{
    public class Kantendetektion
    {
        /// <summary>
        /// Konvertiert das angegebene Bild in Graustufen.
        /// </summary>
        /// <param name="source">Das Eingabebild, das in Graustufen umgewandelt werden soll.</param>
        /// <returns>Das Graustufenbild.</returns>
        public Bitmap Greyscale(Bitmap source)
        {
            Bitmap outputMap = new Bitmap(source);

            for (int y = 0; y < source.Height; y++)
            {
                for (int x = 0; x < source.Width; x++)
                {
                    Color pixelColor = outputMap.GetPixel(x, y);

                    //Wikipedia Formel https://de.wikipedia.org/wiki/Grauwert
                    int grayValue = (int)(0.299 * pixelColor.R + 0.587 * pixelColor.G + 0.114 * pixelColor.B);
                    Color grayColor = Color.FromArgb(grayValue, grayValue, grayValue);
                    outputMap.SetPixel(x, y, grayColor);
                }
            }
            return outputMap;
        }

        /// <summary>
        /// Wendet einen Sobel-Filter zur Kantendetektion auf das Bild an.
        /// </summary>
        /// <param name="source">Das Eingabebild, das gefiltert werden soll.</param>
        /// <returns>Das Bild mit den durch den Sobel-Filter erkannten Kanten.</returns>
        public Bitmap GradientSobelFilter(Bitmap source)
        {
            Bitmap outputMap = Greyscale(source);
            Bitmap finalMap = Greyscale(source);

            int[,] faltungHx = new int[,] {
            {-1, 0, 1 },
            {-2, 0, 2 },
            {-1, 0, 1 }
        };
            int[,] faltungHy = new int[,]
            {
            {-1, -2, -1 },
            {0, 0, 0 },
            {1, 2, 1 }
            };

            for (int y = 1; y < outputMap.Height - 1; y++)
            {
                for (int x = 1; x < outputMap.Width - 1; x++)
                {
                    int sumHx = 0;
                    int I = outputMap.GetPixel(x, y).R;

                    sumHx += outputMap.GetPixel(x - 1, y - 1).R * faltungHx[0, 0];
                    sumHx += outputMap.GetPixel(x, y - 1).R * faltungHx[0, 1];
                    sumHx += outputMap.GetPixel(x + 1, y - 1).R * faltungHx[0, 2];
                    sumHx += outputMap.GetPixel(x - 1, y).R * faltungHx[1, 0];
                    sumHx += I * faltungHx[1, 1];
                    sumHx += outputMap.GetPixel(x + 1, y).R * faltungHx[1, 2];
                    sumHx += outputMap.GetPixel(x - 1, y + 1).R * faltungHx[2, 0];
                    sumHx += outputMap.GetPixel(x, y + 1).R * faltungHx[2, 1];
                    sumHx += outputMap.GetPixel(x + 1, y + 1).R * faltungHx[2, 2];

                    int sumHy = 0;
                    sumHy += outputMap.GetPixel(x - 1, y - 1).R * faltungHy[0, 0];
                    sumHy += outputMap.GetPixel(x, y - 1).R * faltungHy[0, 1];
                    sumHy += outputMap.GetPixel(x + 1, y - 1).R * faltungHy[0, 2];
                    sumHy += outputMap.GetPixel(x - 1, y).R * faltungHy[1, 0];
                    sumHy += I * faltungHy[1, 1];
                    sumHy += outputMap.GetPixel(x + 1, y).R * faltungHy[1, 2];
                    sumHy += outputMap.GetPixel(x - 1, y + 1).R * faltungHy[2, 0];
                    sumHy += outputMap.GetPixel(x, y + 1).R * faltungHy[2, 1];
                    sumHy += outputMap.GetPixel(x + 1, y + 1).R * faltungHy[2, 2];

                    double gradient = Math.Sqrt(Math.Pow(sumHx, 2) + Math.Pow(sumHy, 2));
                    int newvalue = gradient < 150 ? 0 : 255;
                    Color newValueColor = Color.FromArgb(newvalue, newvalue, newvalue);
                    finalMap.SetPixel(x, y, newValueColor);
                }
            }

            return finalMap;
        }

        /// <summary>
        /// Wendet einen Prewitt-Filter zur Kantendetektion auf das Bild an.
        /// </summary>
        /// <param name="source">Das Eingabebild, das gefiltert werden soll.</param>
        /// <returns>Das Bild mit den durch den Prewitt-Filter erkannten Kanten.</returns>
        public Bitmap GradientPrewittFilter(Bitmap source)
        {
            Bitmap outputMap = Greyscale(source);
            Bitmap finalMap = Greyscale(source);

            int[,] faltungHx = new int[,] {
            {-1, 0, 1 },
            {-1, 0, 1 },
            {-1, 0, 1 }
        };
            int[,] faltungHy = new int[,]
            {
            {-1, -1, -1 },
            {0, 0, 0 },
            {1, 1, 1 }
            };

            for (int y = 1; y < outputMap.Height - 1; y++)
            {
                for (int x = 1; x < outputMap.Width - 1; x++)
                {
                    int sumHx = 0;
                    int I = outputMap.GetPixel(x, y).R;

                    sumHx += outputMap.GetPixel(x - 1, y - 1).R * faltungHx[0, 0];
                    sumHx += outputMap.GetPixel(x, y - 1).R * faltungHx[0, 1];
                    sumHx += outputMap.GetPixel(x + 1, y - 1).R * faltungHx[0, 2];
                    sumHx += outputMap.GetPixel(x - 1, y).R * faltungHx[1, 0];
                    sumHx += I * faltungHx[1, 1];
                    sumHx += outputMap.GetPixel(x + 1, y).R * faltungHx[1, 2];
                    sumHx += outputMap.GetPixel(x - 1, y + 1).R * faltungHx[2, 0];
                    sumHx += outputMap.GetPixel(x, y + 1).R * faltungHx[2, 1];
                    sumHx += outputMap.GetPixel(x + 1, y + 1).R * faltungHx[2, 2];

                    int sumHy = 0;
                    sumHy += outputMap.GetPixel(x - 1, y - 1).R * faltungHy[0, 0];
                    sumHy += outputMap.GetPixel(x, y - 1).R * faltungHy[0, 1];
                    sumHy += outputMap.GetPixel(x + 1, y - 1).R * faltungHy[0, 2];
                    sumHy += outputMap.GetPixel(x - 1, y).R * faltungHy[1, 0];
                    sumHy += I * faltungHy[1, 1];
                    sumHy += outputMap.GetPixel(x + 1, y).R * faltungHy[1, 2];
                    sumHy += outputMap.GetPixel(x - 1, y + 1).R * faltungHy[2, 0];
                    sumHy += outputMap.GetPixel(x, y + 1).R * faltungHy[2, 1];
                    sumHy += outputMap.GetPixel(x + 1, y + 1).R * faltungHy[2, 2];

                    double gradient = Math.Sqrt(Math.Pow(sumHx, 2) + Math.Pow(sumHy, 2));
                    int newvalue = gradient < 120 ? 0 : 255;
                    Color newValueColor = Color.FromArgb(newvalue, newvalue, newvalue);
                    finalMap.SetPixel(x, y, newValueColor);
                }
            }
            return finalMap;
        }

        /// <summary>
        /// Wendet den Laplace-Operator zur Kantendetektion auf das Bild an.
        /// </summary>
        /// <param name="source">Das Eingabebild, das gefiltert werden soll.</param>
        /// <returns>Das Bild mit den durch den Laplace-Filter erkannten Kanten.</returns>
        public Bitmap Laplace(Bitmap source)
        {
            Bitmap outputMap = Greyscale(source);
            Bitmap finalMap = Greyscale(source);

            int[,] faltung = new int[,] {
                          { 0, 1, 0},
                          { 1, -4, 1},
                          { 0, 1, 0}
        };
            for (int y = 1; y < source.Height - 1; y++)
            {
                for (int x = 1; x < source.Width - 1; x++)
                {
                    int sum = 0;
                    int I = outputMap.GetPixel(x, y).R;

                    sum += outputMap.GetPixel(x - 1, y - 1).R * faltung[0, 0];
                    sum += outputMap.GetPixel(x, y - 1).R * faltung[0, 1];
                    sum += outputMap.GetPixel(x + 1, y - 1).R * faltung[0, 2];
                    sum += outputMap.GetPixel(x - 1, y).R * faltung[1, 0];
                    sum += I * faltung[1, 1];
                    sum += outputMap.GetPixel(x + 1, y).R * faltung[1, 2];
                    sum += outputMap.GetPixel(x - 1, y + 1).R * faltung[2, 0];
                    sum += outputMap.GetPixel(x, y + 1).R * faltung[2, 1];
                    sum += outputMap.GetPixel(x + 1, y + 1).R * faltung[2, 2];

                    int newValue = sum;
                    newValue = Math.Max(0, Math.Min(255, newValue));
                    int schwellenwert = 40;
                    newValue = newValue > schwellenwert ? 255 : 0;
                    Color newColor = Color.FromArgb(newValue, newValue, newValue);
                    finalMap.SetPixel(x, y, newColor);
                }
            }
            return finalMap;
        }

        /// <summary>
        /// Wendet eine Schärfung mithilfe des Laplace-Filters auf das Graustufenbild an.
        /// </summary>
        /// <param name="source">Das Eingabebild, das geschärft werden soll.</param>
        /// <param name="schärfe">Der Schärfefaktor.</param>
        /// <returns>Das geschärfte Bild.</returns>
        public Bitmap LaplaceSchärfung(Bitmap source, double schärfe)
        {
            Bitmap outputMap1 = Greyscale(source);
            Bitmap finalMap1 = Greyscale(source);

            int[,] faltung1 = new int[,] {
                          { 0, 1, 0},
                          { 1, -4, 1},
                          { 0, 1, 0}
        };
            for (int y = 1; y < source.Height - 1; y++)
            {
                for (int x = 1; x < source.Width - 1; x++)
                {
                    int sum = 0;
                    int I = outputMap1.GetPixel(x, y).R;

                    sum += outputMap1.GetPixel(x - 1, y - 1).R * faltung1[0, 0];
                    sum += outputMap1.GetPixel(x, y - 1).R * faltung1[0, 1];
                    sum += outputMap1.GetPixel(x + 1, y - 1).R * faltung1[0, 2];
                    sum += outputMap1.GetPixel(x - 1, y).R * faltung1[1, 0];
                    sum += I * faltung1[1, 1];
                    sum += outputMap1.GetPixel(x + 1, y).R * faltung1[1, 2];
                    sum += outputMap1.GetPixel(x - 1, y + 1).R * faltung1[2, 0];
                    sum += outputMap1.GetPixel(x, y + 1).R * faltung1[2, 1];
                    sum += outputMap1.GetPixel(x + 1, y + 1).R * faltung1[2, 2];

                    double w = schärfe;
                    int newValue = (int)(I - w * sum);
                    newValue = Math.Max(0, Math.Min(255, newValue));

                    Color newColor = Color.FromArgb(newValue, newValue, newValue);
                    finalMap1.SetPixel(x, y, newColor);
                }
            }
            return finalMap1;
        }

        /// <summary>
        /// Wendet eine Schärfung mithilfe des Laplace-Filters auf das RGB-Bild an.
        /// </summary>
        /// <param name="source">Das Eingabebild, das geschärft werden soll.</param>
        /// <param="schärfe">Der Schärfefaktor.</param>
        /// <returns>Das geschärfte RGB-Bild.</returns>
        public Bitmap LaplaceSchärfungRGB(Bitmap source, double schärfe)
        {
            Bitmap finalMap = new Bitmap(source.Width, source.Height);

            int[,] faltung1 = new int[,] {
        { 0, 1, 0},
        { 1, -4, 1},
        { 0, 1, 0}
    };

            for (int y = 1; y < source.Height - 1; y++)
            {
                for (int x = 1; x < source.Width - 1; x++)
                {
                    // Rot
                    int sumR = 0;
                    int I_R = source.GetPixel(x, y).R;

                    sumR += source.GetPixel(x - 1, y - 1).R * faltung1[0, 0];
                    sumR += source.GetPixel(x, y - 1).R * faltung1[0, 1];
                    sumR += source.GetPixel(x + 1, y - 1).R * faltung1[0, 2];
                    sumR += source.GetPixel(x - 1, y).R * faltung1[1, 0];
                    sumR += I_R * faltung1[1, 1];
                    sumR += source.GetPixel(x + 1, y).R * faltung1[1, 2];
                    sumR += source.GetPixel(x - 1, y + 1).R * faltung1[2, 0];
                    sumR += source.GetPixel(x, y + 1).R * faltung1[2, 1];
                    sumR += source.GetPixel(x + 1, y + 1).R * faltung1[2, 2];

                    int newRed = (int)(I_R - schärfe * sumR);
                    newRed = Math.Max(0, Math.Min(255, newRed));

                    // Grün
                    int sumG = 0;
                    int I_G = source.GetPixel(x, y).G;

                    sumG += source.GetPixel(x - 1, y - 1).G * faltung1[0, 0];
                    sumG += source.GetPixel(x, y - 1).G * faltung1[0, 1];
                    sumG += source.GetPixel(x + 1, y - 1).G * faltung1[0, 2];
                    sumG += source.GetPixel(x - 1, y).G * faltung1[1, 0];
                    sumG += I_G * faltung1[1, 1];
                    sumG += source.GetPixel(x + 1, y).G * faltung1[1, 2];
                    sumG += source.GetPixel(x - 1, y + 1).G * faltung1[2, 0];
                    sumG += source.GetPixel(x, y + 1).G * faltung1[2, 1];
                    sumG += source.GetPixel(x + 1, y + 1).G * faltung1[2, 2];

                    int newGreen = (int)(I_G - schärfe * sumG);
                    newGreen = Math.Max(0, Math.Min(255, newGreen));

                    // Blau
                    int sumB = 0;
                    int I_B = source.GetPixel(x, y).B;

                    sumB += source.GetPixel(x - 1, y - 1).B * faltung1[0, 0];
                    sumB += source.GetPixel(x, y - 1).B * faltung1[0, 1];
                    sumB += source.GetPixel(x + 1, y - 1).B * faltung1[0, 2];
                    sumB += source.GetPixel(x - 1, y).B * faltung1[1, 0];
                    sumB += I_B * faltung1[1, 1];
                    sumB += source.GetPixel(x + 1, y).B * faltung1[1, 2];
                    sumB += source.GetPixel(x - 1, y + 1).B * faltung1[2, 0];
                    sumB += source.GetPixel(x, y + 1).B * faltung1[2, 1];
                    sumB += source.GetPixel(x + 1, y + 1).B * faltung1[2, 2];

                    int newBlue = (int)(I_B - schärfe * sumB);
                    newBlue = Math.Max(0, Math.Min(255, newBlue));

                    Color newColor = Color.FromArgb(newRed, newGreen, newBlue);
                    finalMap.SetPixel(x, y, newColor);
                }
            }
            return finalMap;
        }

        public Bitmap GaussianBlurFilter(Bitmap source, double sigma)
        {
            Bitmap outputMap = Greyscale(source);
            Bitmap finalMap = new Bitmap(source.Width, source.Height);

            int kernelSize = 3;
            int radius = kernelSize / 2;
            double[,] kernel = new double[kernelSize, kernelSize];
            double kernelSum = 0.0;

            for (int y = -radius; y <= radius; y++)
            {
                for (int x = -radius; x <= radius; x++)
                {
                    double value = (1 / (2 * Math.PI * Math.Pow(sigma, 2))) *
                                   Math.Exp(-((x * x + y * y) / (2 * sigma * sigma)));
                    kernel[x + radius, y + radius] = value;
                    kernelSum += value;
                }
            }

            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    kernel[i, j] /= kernelSum;
                }
            }

            for (int y = radius; y < outputMap.Height - radius; y++)
            {
                for (int x = radius; x < outputMap.Width - radius; x++)
                {
                    double sum = 0.0;

                    // Kernel auf die Nachbarpixel anwenden
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            Color pixelColor = outputMap.GetPixel(x + kx, y + ky);
                            sum += pixelColor.R * kernel[kx + radius, ky + radius];
                        }
                    }

                    int newValue = Math.Max(0, Math.Min(255, (int)sum));
                    Color newColor = Color.FromArgb(newValue, newValue, newValue);
                    finalMap.SetPixel(x, y, newColor);
                }
            }

            return finalMap;
        }

        public Bitmap LanczosResampling(Bitmap source, int scale)
        {
            int newWidth = source.Width * scale;
            int newHeight = source.Height * scale;
            Bitmap output = new Bitmap(newWidth, newHeight);

            double a = 3.0; // Die Lanczos-Fenstergröße (3 ist häufig verwendet)

            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    // Ursprüngliche Pixelkoordinaten
                    double srcX = (double)x / scale;
                    double srcY = (double)y / scale;

                    // Rundung auf die nächstgelegene Pixelposition
                    int x0 = (int)Math.Floor(srcX);
                    int y0 = (int)Math.Floor(srcY);

                    // Berechnung des Farbwertes unter Berücksichtigung des Lanczos-Resampling
                    Color color = LanczosPixelValue(source, srcX, srcY, x0, y0, a);
                    output.SetPixel(x, y, color);
                }
            }
            Console.WriteLine("Done");

            return output;
        }

        private Color LanczosPixelValue(Bitmap source, double srcX, double srcY, int x0, int y0, double a)
        {
            double sumR = 0, sumG = 0, sumB = 0, weightSum = 0;

            for (int j = (int)-a + 1; j < (int)a; j++) // Umwandlung hier
            {
                for (int i = (int)-a + 1; i < (int)a; i++) // Umwandlung hier
                {
                    int x = x0 + i;
                    int y = y0 + j;

                    if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                    {
                        double lanczosWeight = LanczosKernel((srcX - x), a) * LanczosKernel((srcY - y), a);
                        Color pixelColor = source.GetPixel(x, y);

                        sumR += pixelColor.R * lanczosWeight;
                        sumG += pixelColor.G * lanczosWeight;
                        sumB += pixelColor.B * lanczosWeight;
                        weightSum += lanczosWeight;
                    }
                }
            }

            // Normalisieren der Farbwerte
            if (weightSum > 0)
            {
                int r = (int)(sumR / weightSum);
                int g = (int)(sumG / weightSum);
                int b = (int)(sumB / weightSum);

                // Manuelle Beschränkung der Farbwerte auf 0-255
                if (r < 0) r = 0; else if (r > 255) r = 255;
                if (g < 0) g = 0; else if (g > 255) g = 255;
                if (b < 0) b = 0; else if (b > 255) b = 255;

                return Color.FromArgb(r, g, b);
            }

            return Color.Black; // Standardfarbe für leere Pixel
        }

        private double LanczosKernel(double x, double a)
        {
            if (x == 0)
                return 1.0;

            if (Math.Abs(x) >= a)
                return 0.0;

            return a * Math.Sin(Math.PI * x) * Math.Sin(Math.PI * x / a) / (Math.PI * Math.PI * x);
        }
    }
}
