using System.Collections;
using System.Collections.Generic;
using lmath;
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

        Assert.NotNull(A.AccessData(), "Empty matrix internal data array is not initialized.");
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

        Assert.NotNull(B.AccessData(), "Empty matrix internal data array is not initialized.");
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

        Assert.NotNull(C.AccessData(), "Empty matrix internal data array is not initialized.");
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
}
