using System;

namespace RechenmethodenMatrix
{
    public class RechenMethoden
    {
        //Addition of 2 matrices
        /// <summary>
        /// matrix1 + matrix2
        /// </summary>
        public double[,] Addition(double[,] matrix1, double[,] matrix2)
        {
            if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1))
            {
                throw new Exception("The matrices are not the same size!");
            }
            int rows = matrix1.GetLength(0);
            int colums = matrix1.GetLength(1);
            double[,] result = new double[rows, colums];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < colums; j++)
                {
                    result[i, j] = matrix1[i, j] + matrix2[i, j];
                }
            }
            return result;
        }

        //Subtraction of 2 matrices
        /// <summary>
        /// matrix1 - matrix2
        /// </summary>
        public double[,] Subtraction(double[,] matrix1, double[,] matrix2)
        {
            if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1))
            {
                throw new Exception("The matrices are not the same size!");
            }
            int rows = matrix1.GetLength(0);
            int colums = matrix1.GetLength(1);
            double[,] result = new double[rows, colums];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < colums; j++)
                {
                    result[i, j] = matrix1[i, j] - matrix2[i, j];
                }
            }
            return result;
        }

        //Multiplication of 2 matrices
        /// <summary>
        /// matrix1 * matrix2
        /// </summary>
        public double[,] Multiplication(double[,] matrix1, double[,] matrix2)
        {
            if (matrix1.GetLength(1) != matrix2.GetLength(0))
            {
                throw new Exception("The matrices cannot be multiplied as they have a different number of rows in matrix 1 and columns in matrix 2!");
            }
            int rows = matrix1.GetLength(0);
            int colums = matrix2.GetLength(1);
            double[,] result = new double[rows, colums];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < colums; j++)
                {
                    for (int k = 0; k < matrix1.GetLength(1); k++)
                    {
                        result[i, j] += matrix1[i, k] * matrix2[k, j];
                    }
                }
            }
            return result;
        }

        //Random Matrices Generating
        /// <summary>
        /// Creates a random Matrice 
        /// </summary>
        public double[,] TestMatrices(int colums, int rows)
        {
            Random rnd = new Random();
            double[,] matrix = new double[rows, colums];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < colums; j++)
                {
                    matrix[i, j] = rnd.NextDouble() * 1000;
                }
            }
            return matrix;
        }

        //Outprint matrices
        /// <summary>
        /// Prints out Matrices in Console
        /// </summary>
        public static void PrintMatrice(double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    Console.Write(matrix[i, j] + "\t");
                }
                Console.WriteLine();
            }
        }

        //Swap Rows
        /// <summary>
        /// Swaps row1 with row2
        /// </summary>
        public double[,] SwapRows(double[,] matrix, int row1, int row2)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            //Check whether there are enough rows (must be at least 2!)
            if (rows <= 2)
            {
                throw new Exception("There are too few rows to swap!");
            }
            //Check whether the 2 interchangeable rows are in the matrix
            if (row1 < 0 || row1 >= rows || row2 < 0 || row2 >= rows)
            {
                throw new Exception("The row index is outside the valid range!");
            }
            double[] staticArray = new double[cols];

            for (int j = 0; j < cols; j++)
            {
                staticArray[j] = matrix[row1, j];
                matrix[row1, j] = matrix[row2, j];
                matrix[row2, j] = staticArray[j];
            }
            return matrix;
        }

        //Row scaling with matrix, exact row and scaling factor
        /// <summary>
        /// Matrix row * factor
        /// </summary>
        public double[,] RowScaling(double[,] matrix, int row, double factor)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            //Condition whether the row series is in the matrix!
            if (row < 0 || row >= rows)
            {
                throw new Exception("The row series is outside the permissible row range of the matrix!");
            }

            for (int j = 0; j < cols; j++)
            {
                matrix[row, j] = matrix[row, j] * factor;
            }
            return matrix;
        }

        //Scaling row addition, where you can apply the SourceRow times factor to the target row
        /// <summary>
        /// Scales a specific row in the matrix with a given factor and adds it to another row.
        /// </summary>
        public double[,] ScalingRowAddition(double[,] matrix, int targetRow, int sourceRow, double factor)
        {
            int cols = matrix.GetLength(1);

            for (int j = 0; j < cols; j++)
            {
                matrix[targetRow, j] += factor * matrix[sourceRow, j];
            }
            return matrix;
        }

        //Row addition without scaling, where you can apply the SourceRow to the target row
        /// <summary>
        ///  Performs a row addition without scaling by adding the "sourceRow" to the "targetRow".
        /// </summary>
        public double[,] RowAddition(double[,] matrix, int targetRow, int sourceRow)
        {
            int cols = matrix.GetLength(1);

            for (int j = 0; j < cols; j++)
            {
                matrix[targetRow, j] += matrix[sourceRow, j];
            }
            return matrix;
        }

        //Get determinant
        /// <summary>
        ///Calculates the determinant of the given square matrix.
        /// </summary>
        public double Determinante(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            //Test if Quadratic Matrix
            if (rows != cols)
            {
                throw new Exception("The Matrix is not a square matrix!");
            }
            if (rows == 1)
            {
                return matrix[0, 0];
            }
            if (rows == 2)
            {
                return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
            }
            double result = 0;

            for (int i = 0; i < cols; i++)
            {
                double[,] subMatrix = SubMatrix(matrix, 0, i);
                //Laplace's theorem (sign)
                if (i % 2 == 0)
                {
                    result += matrix[0, i] * Determinante(subMatrix);
                }
                else
                {
                    result -= matrix[0, i] * Determinante(subMatrix);
                }
            }
            return result;
        }

        //Gives a sub-matrix from a matrix
        /// <summary>
        ///Creates a sub-matrix from the given matrix by removing the specified row and column.
        /// </summary>
        public double[,] SubMatrix(double[,] matrix, int removedRow, int removedCol)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            double[,] submatrix = new double[rows - 1, cols - 1];
            for (int i = 0; i < rows; i++)
            {
                if (i == removedRow) continue;
                for (int j = 0; j < cols; j++)
                {
                    if (j == removedCol) continue;
                    int rowIndexForSubmatrix = i < removedRow ? i : i - 1;
                    int colIndexForSubmatrix = j < removedCol ? j : j - 1;

                    submatrix[rowIndexForSubmatrix, colIndexForSubmatrix] = matrix[i, j];
                }
            }
            return submatrix;
        }

        //Transpose Matrix (Swap row and column)
        /// <summary>
        ///Creates the transposition of the given matrix by swapping rows and columns.
        /// </summary>
        public double[,] Transpose(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] transposedMatrix = new double[cols, rows];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transposedMatrix[j, i] = matrix[i, j];
                }
            }
            return transposedMatrix;
        }

        //Gives the Adjugates of a Matrix
        /// <summary>
        ///Calculates the adjunct of the given square matrix.
        /// </summary>
        public double[,] Adjunct(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            //Condition for the adjugate
            if (rows != cols)
            {
                throw new Exception("The Matrix is not a Square Matrix!");
            }
            double[,] cofactorMatrix = new double[rows, cols];
            //(-1)^(i + j) * Determinante(submatrix) [CofactorMatrix-Formula]
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double[,] submatrix = SubMatrix(matrix, i, j);
                    double determinante = Determinante(submatrix);
                    cofactorMatrix[i, j] = Math.Pow(-1, i + j) * determinante;
                }
            }
            double[,] adjugate = Transpose(cofactorMatrix);
            return adjugate;
        }

        //Method returns the Inverse of a Matrix
        /// <summary>
        ///Calculates the inverse of the given square matrix.
        /// </summary>
        public double[,] Inverse(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            //Conditions for the Inverse
            if (rows != cols)
            {
                throw new Exception("The Matrix is not a Square Matrix!");
            }
            double determinante = Determinante(matrix);
            if (determinante == 0)
            {
                throw new Exception("The determinant is 0 and therefore the matrix has no inverse!");
            }
            //inverse = adjugate / determinante [Inverse-Formular]
            double[,] inverse = new double[rows, cols];
            double[,] adjugate = Adjunct(matrix);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    inverse[i, j] = adjugate[i, j] / determinante; ;
                }
            }
            return inverse;
        }

        // 2 matrices are "divided" with each other (matrix2 is the divisor)
        /// <summary>
        ///Divides two matrices, where 'matrix2' is the divisor.
        /// </summary>
        public double[,] Divide(double[,] matrix1, double[,] matrix2)
        {
            int rows = matrix2.GetLength(0);
            int cols = matrix2.GetLength(1);

            ////Check if Matrix2 is a square Matrix
            if (rows != cols)
            {
                throw new Exception("The Matrix is not a square Matrix!");
            }
            //double[,] dividedMatrix;
            var inverse = Inverse(matrix2);

            //A / B = A * B^(-1) [Divided-Formular]
            var dividedMatrix = Multiplication(matrix1, inverse);
            return dividedMatrix;
        }


        //LGS SOLVERS

        // Solves a system of linear equations using the Gauss-Jordan elimination method
        /// <summary>
        /// Solves a system of linear equations represented as 'Ax = b' using the Gauss-Jordan elimination method.
        /// This method involves creating an extendedMatrix from 'A' and 'b', then performing row operations to transform 'A' into its reduced row echelon form.
        /// The solution vector 'x' is then extracted from the final column of the augmented matrix.
        /// </summary>
        /// <param name="matrix">The matrix 'A' in the system of equations.</param>
        /// <param name="b">The vector 'b' in the system of equations.</param>
        /// <returns>The solution vector 'x'.</returns>
        public double[] GaussJordan(double[,] matrix, double[] b)
        {
            int n = matrix.GetLength(0);

            double[,] extendedMatrix = new double[n, n + 1];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    extendedMatrix[i, j] = matrix[i, j];
                }
                extendedMatrix[i, n] = b[i];
            }


            for (int i = 0; i < n; i++)
            {
                extendedMatrix = RowScaling(extendedMatrix, i, 1.0 / extendedMatrix[i, i]);

                for (int j = 0; j < n; j++)
                {
                    if (i != j)
                    {
                        // Subtracts the i-fold of the i-th line from the j-th line
                        extendedMatrix = ScalingRowAddition(extendedMatrix, j, i, -extendedMatrix[j, i]);
                    }
                }
            }

            // Extracts vector x (solution of the linear equation system) from the expanded matrix
            double[] x = new double[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = extendedMatrix[i, n];
            }

            return x;
        }

        // Solves a system of linear equations using Cramer's Rule
        /// <summary>
        /// Solves a system of linear equations represented as 'Ax = b' using Cramer's Rule.
        /// This method involves calculating the determinant of the matrix 'A' and then replacing each column of 'A' with the vector 'b' one by one,
        /// calculating the determinant of the resulting matrix, and dividing it by the determinant of 'A' to find each element of the solution vector 'x'.
        /// </summary>
        /// <param name="matrix">The matrix 'A' in the system of equations.</param>
        /// <param name="b">The vector 'b' in the system of equations.</param>
        /// <returns>The solution vector 'x'.</returns>
        /// <exception cref="Exception">Thrown when the matrix is not invertible (i.e., its determinant is zero).</exception>
        public double[] DeterminantMethod(double[,] matrix, double[] b)
        {
            int n = matrix.GetLength(0);
            double[] x = new double[n];
            double determinant = Determinante(matrix);

            for (int i = 0; i < n; i++)
            {
                double[,] cloneMatrix = (double[,])matrix.Clone();

                // Replace the i-th column of the cloneMatrix with the vector b.
                for (int j = 0; j < n; j++)
                {
                    cloneMatrix[j, i] = b[j];
                }
                // Compute the determinant of the modified matrix and store the result in x.
                x[i] = Determinante(cloneMatrix) / determinant;
            }
            return x;
        }

        // Solves a system of linear equations using the inverse of the matrix
        /// <summary>
        /// Solves a system of linear equations represented as 'Ax = b' by using the inverse of the matrix 'A'.
        /// This is achieved by calculating the inverse of 'A' and then multiplying it with the vector 'b' to find the solution vector 'x'.
        /// </summary>
        /// <param name="matrix">The matrix 'A' in the system of equations.</param>
        /// <param name="b">The vector 'b' in the system of equations.</param>
        /// <returns>The solution vector 'x'.</returns>
        /// <exception cref="Exception">Thrown when the matrix is not invertible.</exception>
        public double[] InverseMethod(double[,] matrix, double[] b)
        {
            int rows = matrix.GetLength(0);
            double[,] inverse = Inverse(matrix);
            double[] x = new double[rows];

            for (int i = 0; i < rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < rows; j++)
                {
                    sum += inverse[i, j] * b[j];
                }
                x[i] = sum;
            }
            return x;
        }


        // Sigmoid-Aktivierungsfunktion
        public double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}