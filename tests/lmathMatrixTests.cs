using System.Collections;
using System.Collections.Generic;
using lmath;
using statistics;
using NUnit.Framework;

[TestFixture]
public class lmathMatrixTests {
    Matrix A;
    Matrix B;
    Matrix C;
    Matrix Zeroes;
    Matrix Ones;
    Matrix I;

    [SetUp]
    public void SetUp() {
        A = new Matrix(new float[] {1,2,3,4,5,6,7,8,9}, 3, 3);
        B = new Matrix(new float[,] {{1,2,1}, {2,4,6}, {7,2,5}});
        //AxB = {{26,16,28}, {56,40,64}, {86,64,100}}
        C = new Matrix();
        Zeroes = Matrix.Zeros(3, 3);
        Ones = Matrix.Ones(3, 3);
        I = Matrix.Identity(3);
    }

    // CONSTRUCTORS
    //------------------------------------------------------------------------------
    [Test, Category("Constructor")]
    public void Constructor_Empty() {
        Assert.NotNull(C.AccessData(), "Empty matrix internal data array is not initialized.");
        Assert.AreEqual(2, C.rank, "Empty matrix does not have rank 2.");
        Assert.AreEqual(0, C.GetLength(), "Empty matrix does not have length 0.");
    }

    [Test, Category("Constructor")]
    public void Constructor_FloatArray() {
        float[,] elements = {{1,2,3}, {4,5,6}, {7,8,9}};
        int n = elements.GetLength(0);
        int m = elements.GetLength(1);

        Assert.NotNull(A.AccessData(), "Matrix internal data array is not initialized.");
        Assert.AreEqual(n, A.GetShape()[0], $"Matrix row number {A.GetShape()[0]} does not equal expected row number {n}.");
        Assert.AreEqual(m, A.GetShape()[1], $"Matrix column number {A.GetShape()[1]} does not equal expected column number {m}.");
        Assert.AreEqual(elements.Length, A.GetLength(), $"Matrix length {A.GetLength()} does not equal expected length {elements.Length}.");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                float value = A[i, j];
                float expected = elements[i, j];
                Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {j}] does not equal expected value {expected}.");
            }
        }
    }

    [Test, Category("Constructor")]
    public void Constructor_DoubleFloatArray() {
        float[,] elements = {{1,2,1}, {2,4,6}, {7,2,5}};
        int n = elements.GetLength(0);
        int m = elements.GetLength(1);

        Assert.NotNull(B.AccessData(), "Matrix internal data array is not initialized.");
        Assert.AreEqual(n, B.GetShape()[0], $"Matrix row number {B.GetShape()[0]} does not equal expected row number {n}.");
        Assert.AreEqual(m, B.GetShape()[1], $"Matrix column number {B.GetShape()[1]} does not equal expected column number {m}.");
        Assert.AreEqual(elements.Length, B.GetLength(), $"Matrix length {B.GetLength()} does not equal expected length {elements.Length}.");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                float value = B[i, j];
                float expected = elements[i, j];
                Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {j}] does not equal expected value {expected}.");
            }
        }
    }

    [Test, Category("Constructor")]
    public void Constructor_Matrix() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        int length = A.GetLength();
        C = new Matrix(A);

        Assert.NotNull(C.AccessData(), "Matrix internal data array is not initialized.");
        Assert.AreNotSame(A.AccessData(), C.AccessData(), "Copy constructor did not create a deep copy of internal data.");
        Assert.AreEqual(n, C.GetShape()[0], $"Matrix row number {C.GetShape()[0]} does not equal expected row number {n}.");
        Assert.AreEqual(m, C.GetShape()[1], $"Matrix column number {C.GetShape()[1]} does not equal expected column number {m}.");
        Assert.AreEqual(length, C.GetLength(), $"Matrix length {C.GetLength()} does not equal expected length {length}.");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                float value = C[i, j];
                float expected = A[i, j];
                Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {j}] does not equal expected value {expected}.");
            }
        }
    }

    // Data Management
    //------------------------------------------------------------------------------
    [Test, Category("Data Management")]
    public void DataManagement_GetElement() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        float[,] elements = {{1,2,3}, {4,5,6}, {7,8,9}};
        for(int i = 0; i < n; i++) { 
            for(int j = 0; j < m; j++) {
                float value = A.GetElement(i, j);
                float expected = elements[i, j];
                Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {j}] does not equal expected value {expected}.");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_SetElement() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        C = Matrix.Zeros(n, m);
        for(int i = 0; i < n; i++) { 
            for(int j = 0; j < m; j++) {
                C.SetElement(A[i, j], i, j);
                for(int i2 = 0; i2 < n; i2++) {
                    for(int j2 = 0; j2 < m; j2++) { 
                        float value = C[i2, j2];
                        float expected = 0.0f;
                        if(i == i2 && j == j2) {
                            expected = A[i2, j2];
                        }
                        Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {j}] does not equal expected value {expected}.");
                    }
                }

                C = Matrix.Zeros(m, n);
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_GetRow() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        for(int i = 0; i < n; i++) {
            C = A.GetRow(i);
            Assert.AreEqual(1, C.GetShape()[0], $"Row matrix shape[{0}] {C.GetShape()[0]} does not equal expected value {1}.");
            Assert.AreEqual(m, C.GetShape()[1], $"Row matrix shape[{1}] {C.GetShape()[1]} does not equal expected value {m}.");
            for(int j = 0; j < m; j++) {
                float value = C[0, j];
                float expected = A[i, j];
                Assert.AreEqual(expected, value, $"Element {value} at index [{0}, {j}] does not equal expected value {expected} at index [{i}, {j}].");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_GetColumn() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        for(int j = 0; j < m; j++) {
            C = A.GetColumn(j);
            Assert.AreEqual(n, C.GetShape()[0], $"Row matrix shape[{0}] {C.GetShape()[0]} does not equal expected value {n}.");
            Assert.AreEqual(1, C.GetShape()[1], $"Row matrix shape[{1}] {C.GetShape()[1]} does not equal expected value {1}.");
            for(int i = 0; i < m; i++) {
                float value = C[i, 0];
                float expected = A[i, j];
                Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {0}] does not equal expected value {expected} at index [{i}, {j}].");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_Reshape_Matrix_GreaterSize() {
        int k = 10;
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        C = new Matrix(A);
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < k; j++) {
                if(i == 0 && j == 0) {
                    continue;
                }

                C.Reshape(n + i, m + j);

                Assert.AreEqual(n + i, C.GetShape()[0], $"Matrix row number {C.GetShape()[0]} does not equal expected row number {n + i}");
                Assert.AreEqual(m + j, C.GetShape()[1], $"Matrix column number {C.GetShape()[1]} does not equal expected column number {m + j}");
                for(int i2 = 0; i2 < n + i; i2++) {
                    for(int j2 = 0; j2 < m + j; j2++) {
                        float value = C[i2, j2];
                        float expected = 0.0f;
                        if(i2 <  n && j2 < m) {
                            expected = A[i2, j2];
                        }
                        
                        Assert.AreEqual(expected, value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {expected}.");
                    }
                }

                C = new Matrix(A);
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_Reshape_Matrix_SameSize() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        C = new Matrix(A);
        C.Reshape(n, m);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                float value = C[i, j];
                float expected = A[i, j];
                Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {j}] does not equal expected value {expected}.");
            }
        }
    }

    [Test, Category("Data Management")]
    public void DataManagement_Reshape_Matrix_SmallerSize() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        C = new Matrix(A);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                if(i == 0 && j == 0) {
                    continue;
                }

                C.Reshape(n - i, m - j);

                Assert.AreEqual(n - i, C.GetShape()[0], $"Matrix row number {C.GetShape()[0]} does not equal expected row number {n - i}");
                Assert.AreEqual(m - j, C.GetShape()[1], $"Matrix column number {C.GetShape()[1]} does not equal expected column number {m - j}");
                for(int i2 = 0; i2 < n - i; i2++) {
                    for(int j2 = 0; j2 < m - j; j2++) {
                        float value = C[i2, j2];
                        float expected = A[i2, j2];
                        
                        Assert.AreEqual(expected, value, $"Element {value} at index [{i2}, {j2}] does not equal expected value {expected}.");
                    }
                }

                C = new Matrix(A);
            }
        }
    }

    // DEFAULT OBJECTS
    //------------------------------------------------------------------------------
    [Test, Category("Default Objects")]
    public void DefaultObject_Zeros() {
        int n = 3;
        int m = 3;
        Assert.AreEqual(n, Zeroes.GetShape()[0], $"Matrix row number {Zeroes.GetShape()[0]} is not expected row number {n}");
        Assert.AreEqual(m, Zeroes.GetShape()[1], $"Matrix column number {Zeroes.GetShape()[1]} is not expected column number {m}");

        for (int i = 0; i < n; i++) {
            for(int j = 0;j < m; j++) {
                Assert.AreEqual(0f, Zeroes[i, j], $"Element {Zeroes[i, j]} at index [{i}, {j}] does not equal expected element {0f}");
            }
        }
    }

    [Test, Category("Default Objects")]
    public void DefaultObject_Ones() {
        int n = 3;
        int m = 3;
        Assert.AreEqual(n, Ones.GetShape()[0], $"Matrix row number {Ones.GetShape()[0]} is not expected row number {n}");
        Assert.AreEqual(m, Ones.GetShape()[1], $"Matrix column number {Ones.GetShape()[1]} is not expected column number {m}");

        for (int i = 0; i < n; i++) {
            for(int j = 0;j < m; j++) {
                Assert.AreEqual(1f, Ones[i, j], $"Element {Ones[i, j]} at index [{i}, {j}] does not equal expected element {1f}");
            }
        }
    }

    [Test, Category("Default Objects")]
    public void DefaultObject_Identity() {
        int n = 3;
        Assert.AreEqual(n, I.GetShape()[0], $"Matrix row number {I.GetShape()[0]} is not expected row number {n}");
        Assert.AreEqual(n, I.GetShape()[1], $"Matrix column number {I.GetShape()[1]} is not expected column number {n}");

        for (int i = 0; i < n; i++) {
            for(int j = 0;j < n; j++) {
                float value = I[i, j];
                float expected = 0.0f;
                if(i == j) {
                    expected = 1.0f;
                }
                Assert.AreEqual(expected, value, $"Element {value} at index [{i}, {j}] does not equal expected element {expected}");
            }
        }
    }

    [Test, Category("Default Objects")]
    public void DefaultObject_Diagonal_Element() {
        int e = 5;
        int n = 10;
        int[] ra = {1, n};
        int[] rb = {1, n};
        for(int i = ra[0]; i < ra[1]; i++) {
            for(int j = rb[0]; j < rb[1]; j++) {
                C = Matrix.Diag(e, i, j);
                Assert.AreEqual(i, C.GetShape()[0], $"Matrix row number {C.GetShape()[0]} is not expected row number {i}");
                Assert.AreEqual(j, C.GetShape()[1], $"Matrix column number {C.GetShape()[1]} is not expected column number {j}");

                for(int i2 = 0; i2 < i; i2++) { 
                    for(int j2 = 0; j2 < j; j2++) {
                        float value = C[i2, j2];
                        float expected = 0.0f;
                        if(i2 == j2) {
                            expected = e;
                        }
                        Assert.AreEqual(expected, value, $"Element {value} at index [{i2}, {j2}] does not equal expected element {expected}");
                    }
                }
            }
        }
    }

    [Test, Category("Default Objects")]
    public void DefaultObject_Diagonal_Data() {
        float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        int n = data.Length;
        int[] ra = {1, n};
        int[] rb = {1, n};
        for(int i = ra[0]; i < ra[1]; i++) {
            for(int j = rb[0]; j < rb[1]; j++) {
                float[] temp = new float[System.Math.Min(i, j)];
                for (int k = 0; k < temp.Length; k++) {
                    temp[k] = data[k];
                }
                C = Matrix.Diag(temp, i, j);
                Assert.AreEqual(i, C.GetShape()[0], $"Matrix row number {C.GetShape()[0]} is not expected row number {i}");
                Assert.AreEqual(j, C.GetShape()[1], $"Matrix column number {C.GetShape()[1]} is not expected column number {j}");

                for(int i2 = 0; i2 < i; i2++) { 
                    for(int j2 = 0; j2 < j; j2++) {
                        float value = C[i2, j2];
                        float expected = 0.0f;
                        if(i2 == j2) {
                            expected = data[i2];
                        }
                        Assert.AreEqual(expected, value, $"Element {value} at index [{i2}, {j2}] does not equal expected element {expected}");
                    }
                }
            }
        }
    }

    // OPERATIONS
    //------------------------------------------------------------------------------
    [Test, Category("Operations")]
    public void Operations_Scale() {
        float expected = 2.0f;
        int n = Ones.GetShape()[0];
        int m = Ones.GetShape()[1];
        Ones.Scale(expected);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                Assert.That(Ones[i, j], Is.EqualTo(expected).Within(1e-5f), $"Element {Ones[i, j]} at index [{i}, {j}] does not equal expected element {expected}");
            }
        }
    }

    [Test, Category("Operations")]
    public void Operations_Negate() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];
        C = Matrix.Negate(A);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                Assert.That(C[i, j], Is.EqualTo(-A[i, j]).Within(1e-5f), $"Element {C[i, j]} at index [{i}, {j}] does not equal expected element {-A[i, j]}");
            }
        }
    }

    [TestCase("Add"), Category("Operations")]
    [TestCase("Subtract"), Category("Operations")]
    [TestCase("Hadamard"), Category("Operations")]
    public void Operations_TwoMatrices_ElementWise(string operation) {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];

        switch (operation) {
            case "Add": C = A + B; break;
            case "Subtract": C = A - B; break;
            case "Hadamard": C = Matrix.HadamardProduct(A, B); break;
            default: throw new System.ArgumentException();
        }

        for(int i = 0; i < n; i++) {
            for( int j = 0; j < m; j++) {
                float expected = operation switch {
                    "Add" => A[i, j] + B[i, j],
                    "Subtract" => A[i, j] - B[i, j],
                    "Hadamard" => A[i, j] * B[i, j],
                    _ => 0
                };

                Assert.That(C[i, j], Is.EqualTo(expected).Within(1e-5f), $"Element {C[i, j]} at index [{i}, {j}] does not equal expected element {expected}");
            }
        }
    }

    [Test, Category("Operations")]
    public void Operations_MatMul() {
        float[,] expected = {{26,16,28}, {56,40,64}, {86,64,100}};
        int n = expected.GetLength(0);
        int m = expected.GetLength(1);
        C = Matrix.MatMul(A, B);

        Assert.AreEqual(n, C.GetShape()[0], $"Matrix row number {C.GetShape()[0]} is not expected row number {n}");
        Assert.AreEqual(m, C.GetShape()[1], $"Matrix column number {C.GetShape()[1]} is not expected column number {m}");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                float value = C[i, j];
                Assert.AreEqual(expected[i,j], value, $"Element {value} at index [{i}, {j}] does not equal expected element {expected}");
            }
        }

        //Result Shape Check
        n = 10;
        m = 10;
        int p = 10;
        int[] nrange = {1, n};
        int[] mrange = {1, m};
        int[] prange = {1, p};
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                for(int k = 0; k < p; k++) {
                    A = Matrix.Zeros(i, j);
                    B = Matrix.Zeros(j, k);
                    C = Matrix.MatMul(A, B);
                    Assert.AreEqual(i, C.GetShape()[0], $"Matrix row number {C.GetShape()[0]} is not expected row number {i}");
                    Assert.AreEqual(k, C.GetShape()[1], $"Matrix column number {C.GetShape()[1]} is not expected column number {k}");
                }
            }
        }
    }

    [Test, Category("Operations")]
    public void Operations_MatVecMul() {
        Vector v = new Vector(new float[] {1, 2 ,3});
        float[] expected = {14f, 32f, 50f};
        int n = expected.Length;
        Vector result = Matrix.MatVecMul(A, v);

        Assert.AreEqual(n, result.length, $"Vector length {result.length} not equal to expected length {n}.");
        for(int i = 0; i < n; i++) {
            float value = result[i];
            Assert.That(value, Is.EqualTo(expected[i]).Within(1e-5f), $"Element {value} at index {i} does not equal expected value {expected}.");
        }

        //Result Shape Check
        n = 10;
        int m = 10;
        int[] nrange = {1, n};
        int[] mrange = {1, m};
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                A = Matrix.Zeros(i, j);
                Vector b = Vector.Zeros(j);
                Vector c = Matrix.MatVecMul(A, b);
                Assert.AreEqual(i, c.length, $"Vector length {c.length} is not expected length {i}.");
            }
        }
    }

    [Test, Category("Operations")]
    public void Operations_Transpose() { 
        C = Matrix.Transpose(A);
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];

        Assert.AreEqual(m, C.GetShape()[0], $"Matrix row number {C.GetShape()[0]} is not expected row number {m}");
        Assert.AreEqual(n, C.GetShape()[1], $"Matrix column number {C.GetShape()[1]} is not expected column number {n}");
        for (int i = 0; i < n; i++) { 
            for(int j = 0; j < m; j++) {
                float expected = A[i, j];
                float value = C[j, i];
                Assert.AreEqual(expected, value, $"Element {value} at index [{j}, {i}] does not equal expected element {expected}");
            }
        }
    }

    [Test, Category("Operations")]
    public void Operations_FrobeniusNorm() {
        float expected = System.MathF.Sqrt(16.882f);
        float value = A.Norm();
        Assert.That(value, Is.EqualTo(expected).Within(1e-5f));
    }

    [Test, Category("Operations")]
    public void Operations_Trace() {
        float expected = 15f;
        float value = A.Trace();
        Assert.That(value, Is.EqualTo(expected).Within(1e-5f));
    }

    [Test, Category("Operations")]
    public void Operations_Determinant() {
        float expectedA = 0f;
        float expectedB = 48f;
        float valueA = A.Determinant();
        float valueB = B.Determinant();
        Assert.That(valueA, Is.EqualTo(expectedA).Within(1e-5f));
        Assert.That(valueB, Is.EqualTo(expectedB).Within(1e-5f));
    }

    // RANDOM
    //------------------------------------------------------------------------------
    [Test, Category("Random")]
    public void Random_Uniform_MatrixFromFloats() {
        int bcount = 100; //Bucket Count
        int scount = 100000; //Sample Count

        Matrix sum = Matrix.Zeros(3, 3);
        Vector[] buckets = new Vector[3 * 3] {Vector.Zeros(bcount), Vector.Zeros(bcount), Vector.Zeros(bcount),
                                                Vector.Zeros(bcount), Vector.Zeros(bcount), Vector.Zeros(bcount),
                                                Vector.Zeros(bcount), Vector.Zeros(bcount), Vector.Zeros(bcount)};
        for(int i = 0; i < scount; i++) {
            C = Matrix.Random(0, 1, 3, 3);
            for(int j = 0; j < 9; j++) {
                Assert.That(C[j], Is.InRange(0, 1));
                int bucket = (int)System.Math.Min(bcount-1, System.Math.Floor(C[j] * bcount));
                buckets[j][bucket]++;
            }
            sum += C;
        }

        Matrix mean = sum * (1f / scount);
        float expectedMean = (1f + 0f) / 2f;
        for (int i = 0; i < 9; i++) {
            Assert.That(mean[i], Is.EqualTo(expectedMean).Within(1e-2f));
        }

        int expectedCount = scount / bcount;
        double p = 1.0 / bcount;
        double sigma = System.Math.Sqrt(scount * p * (1-p));
        int expectedMin = expectedCount - (int)(4 * sigma);
        int expectedMax = expectedCount + (int)(4 * sigma);
        for (int i = 0; i < bcount; i++) {
            for(int j = 0; j < 9; j++) {
                Assert.That(buckets[j][i], Is.InRange(expectedMin, expectedMax));
            }
        }
    }

    [Test, Category("Random")]
    public void Random_Normal_MatrixFromFloats() {
        const int bcount = 3; //Bucket Count
        int scount = 100000; //Sample Count
        float expectedMean = 1f;
        float expectedSTDDev = 1f;
        Vector bvalues = Vector.Zeros(bcount);
        for(int i = 0; i < bcount; i++) {
            double cdf = Statistics.NormalCDF(0.0, 1.0, i+1);
            double probability = 2 * cdf - 1;
            bvalues[i] = (float) probability;

        }

        for(int i = bcount - 1; i > 0; i--) {
            bvalues[i] -= bvalues[i - 1];
        }

        Matrix sum = Matrix.Zeros(3, 3);
        Matrix sumSQ = Matrix.Zeros(3, 3);
        Vector[] buckets = new Vector[3 * 3] {Vector.Zeros(bcount), Vector.Zeros(bcount), Vector.Zeros(bcount),
                                                Vector.Zeros(bcount), Vector.Zeros(bcount), Vector.Zeros(bcount),
                                                Vector.Zeros(bcount), Vector.Zeros(bcount), Vector.Zeros(bcount)};
        for (int i = 0; i < scount; i++) {
            C = Matrix.RandomN(expectedMean, expectedSTDDev, 3, 3);
            sum += C;
            for(int j = 0; j < 9; j++) {
                sumSQ[j] += C[j] * C[j];
            }

            for(int j = 0; j < 9; j++) {
                bool bucketFound = false;
                float value = System.MathF.Abs(C[j] - expectedMean);
                for(int k = 0; k < bcount-1; k++) {
                    if(value <= (k+1) * expectedSTDDev) {
                        buckets[j][k]++;
                        bucketFound = true;
                        break;
                    }
                }

                if(!bucketFound) {
                    buckets[j][bcount - 1]++;
                }
            }
        }

        Matrix mean = sum * (1f / scount);
        for(int i = 0; i < 9; i++) {
            Assert.That(mean[i], Is.EqualTo(expectedMean).Within(1e-2f));
        }

        Matrix variance = (sumSQ * (1f / scount)) - Matrix.HadamardProduct(mean, mean);
        Matrix stdDev = new Matrix(variance);
        for(int i = 0; i < 9; i++) {
            stdDev[i] = System.MathF.Sqrt(stdDev[i]);
            Assert.That(stdDev[i], Is.EqualTo(expectedSTDDev).Within(1e-2f));
        }
        
        for(int i = 0; i < bcount; i++) {
            for (int j = 0; j < 9; j++) {
                float percent = (float)buckets[j][i]/scount;
                Assert.That(percent, Is.EqualTo(bvalues[i]).Within(1e-2f));
            }
        }
    }

    //CONDITIONS
    //------------------------------------------------------------------------------
    [Test, Category("Conditions")]
    public void Conditions_Symmetric() {
        int n = A.GetShape()[0];
        int m = A.GetShape()[1];

        C = new Matrix(A);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                C[i, j] = C[j, i];
            }
        }

        Assert.IsFalse(A.IsSymmetric());
        Assert.IsTrue(Zeroes.IsSymmetric());
        Assert.IsTrue(C.IsSymmetric());
    }
}
