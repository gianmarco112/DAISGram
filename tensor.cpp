#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <fstream>

#include "dais_exc.h"
#include "tensor.h"

#define PI 3.141592654
#define FLT_MAX 3.402823466e+38F /* max value */
#define FLT_MIN 1.175494351e-38F /* min positive value */

using namespace std;

/**
     * Class constructor
     *
     * Parameter-less class constructor
     */
Tensor::Tensor() { 
    data = nullptr;
    r = 0;
    c = 0;
    d = 0;
}

/**
 * Class constructor
 *
 * Creates a new tensor of size r*c*d initialized at value v
 *
 * @param r
 * @param c
 * @param d
 * @param v
 * @return new Tensor
 */
Tensor::Tensor(int r, int c, int d, float v){
    this->r = r;
    this->c = c;
    this->d = d;

    data = new float** [r];

    for (int i = 0; i < r; ++i){

        data[i] = new float* [c];

        for (int j = 0; j < c; ++j){

            data[i][j] = new float[d];

            for (int k = 0; k < d; ++k){
                data[i][j][k] = v;
            }
        }
    }
}



/**
 * Copy constructor
 *
 * This constructor copies the data from another Tensor
 *
 * @return the new Tensor
 */
Tensor::Tensor(const Tensor& that){
    r = that.r;
    c = that.c;
    d = that.d;

    data = new float** [r];

    for (int i = 0; i < r; ++i){

        data[i] = new float* [c];

        for (int j = 0; j < c; ++j){

            data[i][j] = new float[d];

            for (int k = 0; k < d; ++k){
                data[i][j][k] = that(i, j, k);
            }
        }
    }
}
/**
* Class distructor
*
 * Cleanup the data when deallocated
 */
Tensor::~Tensor(){
    for (int i = 0; i < r; ++i){
        for (int j = 0; j < c; ++j){
            delete[] data[i][j];
        }
        delete[] data[i];
    }
    delete[] data;
}

/**
     * Operator overloading ()
     *
     * if indexes are out of bound throw index_out_of_bound() exception
     *
     * @return the value at location [i][j][k]
     */
float Tensor::operator()(int i, int j, int k) const{

    if (data == nullptr) throw (tensor_not_initialized());

    if (i >= r || i < 0) throw (index_out_of_bound());
    if (j >= c || j < 0) throw (index_out_of_bound());
    if (k >= d || k < 0) throw (index_out_of_bound());

    return data[i][j][k];
}

/**
 * Operator overloading ()
 *
 * Return the pointer to the location [i][j][k] such that the operator (i,j,k) can be used to
 * modify tensor data.
 *
 * If indexes are out of bound throw index_out_of_bound() exception
 *
 * @return the pointer to the location [i][j][k]
 */
float& Tensor::operator()(int i, int j, int k){

    if (data == nullptr) throw (tensor_not_initialized());

    if (i >= r || i < 0) throw (index_out_of_bound());
    if (j >= c || j < 0) throw (index_out_of_bound());
    if (k >= d || k < 0) throw (index_out_of_bound());

    return data[i][j][k];
}

/**
 * Operator overloading ==
 *
 * It performs the point-wise equality check between two Tensors.
 *
 * The equality check between floating points cannot be simply performed using the
 * operator == but it should take care on their approximation.
 *
 * This approximation is known as rounding (do you remember "Architettura degli Elaboratori"?)
 *
 * For example, given a=0.1232 and b=0.1233 they are
 * - the same, if we consider a rounding with 1, 2 and 3 decimals
 * - different when considering 4 decimal points. In this case b>a
 *
 * So, given two floating point numbers "a" and "b", how can we check their equivalence?
 * through this formula:
 *
 * a == b if and only if |a-b|<EPSILON
 *
 * where EPSILON is fixed constant (defined at the beginning of this header file)
 *
 * Two tensors A and B are the same if:
 * A[i][j][k] == B[i][j][k] for all i,j,k
 * where == is the above formula.
 *
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 *
 * @return returns true if all their entries are "floating" equal
 */
bool Tensor::operator==(const Tensor& rhs) const{

    if (data == nullptr) throw (tensor_not_initialized());

    if (rhs.data == nullptr) throw (tensor_not_initialized());

    if (r != rhs.r || c != rhs.c || d != rhs.d) throw (dimension_mismatch());

    bool check = true;

    int i = 0, j = 0, k = 0;

    while (i < r && check){

        j = 0;

        while (j < c && check){

            k = 0;

            while (k < d && check){
                float delta = data[i][j][k] - rhs(i, j, k);

                if (abs(delta) >= EPSILON) check = false;

                ++k;
            }
            ++j;
        }
        ++i;
    }

    return check;
}

/**
 * Operator overloading -
 *
 * It performs the point-wise difference between two Tensors.
 *
 * result(i,j,k)=this(i,j,k)-rhs(i,j,k)
 *
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 *
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator-(const Tensor& rhs)const{

    if (data == nullptr)     throw (tensor_not_initialized());

    if (rhs.data == nullptr) throw (tensor_not_initialized());
    
    if (r != rhs.r || c != rhs.c || d != rhs.d) throw (dimension_mismatch());


    Tensor copy(*this);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                copy(i, j, k) -= rhs(i, j, k);
            }
        }
    }

    return copy;
}

 /**
 * Operator overloading +
 *
 * It performs the point-wise sum between two Tensors.
 *
 * result(i,j,k)=this(i,j,k)+rhs(i,j,k)
 *
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 *
 * @return returns a new Tensor containing the result of the operation
*/
Tensor Tensor::operator +(const Tensor& rhs)const{

    if (data == nullptr) throw (tensor_not_initialized());

    if (rhs.data == nullptr) throw (tensor_not_initialized());

    if (r != rhs.r || c != rhs.c || d != rhs.d) throw (dimension_mismatch());

    Tensor copy(*this);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                copy(i, j, k) += rhs(i, j, k);
            }
        }
    }

    return copy;
}

/**
 * Operator overloading *
 *
 * It performs the point-wise product between two Tensors.
 *
 * result(i,j,k)=this(i,j,k)*rhs(i,j,k)
 *
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 *
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator*(const Tensor& rhs)const{

    if (data == nullptr) throw (tensor_not_initialized());

    if (rhs.data == nullptr) throw (tensor_not_initialized());

    if (r != rhs.r || c != rhs.c || d != rhs.d) throw (dimension_mismatch());

    Tensor copy(*this);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                copy(i, j, k) *= rhs(i, j, k);
            }
        }
    }

    return copy;
}

/**
 * Operator overloading /
 *
 * It performs the point-wise division between two Tensors.
 *
 * result(i,j,k)=this(i,j,k)/rhs(i,j,k)
 *
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 *
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator/(const Tensor& rhs)const{

    if (data == nullptr) throw (tensor_not_initialized());

    if (rhs.data == nullptr) throw (tensor_not_initialized());

    if (r != rhs.r || c != rhs.c || d != rhs.d) throw (dimension_mismatch());

    Tensor copy(*this);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                copy(i, j, k) /= rhs(i, j, k);
            }
        }
    }

    return copy;
}

/**
 * Operator overloading -
 *
 * It performs the point-wise difference between a Tensor and a constant
 *
 * result(i,j,k)=this(i,j,k)-rhs
 *
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator-(const float& rhs)const{

    if (data == nullptr) throw (tensor_not_initialized());

    Tensor copy(*this);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                copy(i, j, k) -= rhs;
            }
        }
    }

    return copy;
}

/**
 * Operator overloading +
 *
 * It performs the point-wise sum between a Tensor and a constant
 *
 * result(i,j,k)=this(i,j,k)+rhs
 *
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator+(const float& rhs)const{

    if (data == nullptr) throw (tensor_not_initialized());

    Tensor copy(*this);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                copy(i, j, k) += rhs;
            }
        }
    }

    return copy;
}

/**
 * Operator overloading *
 *
 * It performs the point-wise product between a Tensor and a constant
 *
 * result(i,j,k)=this(i,j,k)*rhs
 *
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator*(const float& rhs)const{

    if (data == nullptr) throw (tensor_not_initialized());

    Tensor copy(*this);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                copy(i, j, k) *= rhs;
            }
        }
    }

    return copy;
}

/**
 * Operator overloading / between a Tensor and a constant
 *
 * It performs the point-wise division between a Tensor and a constant
 *
 * result(i,j,k)=this(i,j,k)/rhs
 *
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator/(const float& rhs)const{

    if (data == nullptr) throw (tensor_not_initialized());

    if (rhs == 0) throw (unknown_operation());

    Tensor copy(*this);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                copy(i, j, k) /= rhs;
            }
        }
    }

    return copy;
}

/**
 * Operator overloading = (assignment)
 *
 * Perform the assignment between this object and another
 *
 * @return a reference to the receiver object
 */
Tensor& Tensor::operator=(const Tensor& other){

    if (other.data == nullptr) throw (tensor_not_initialized());

    init(other.r, other.c, other.d, 0.0f);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                data[i][j][k] = other(i, j, k);
            }
        }
    }

    return *this;
}

/**
 * Random Initialization
 *
 * Perform a random initialization of the tensor
 *
 * @param mean The mean
 * @param std  Standard deviation
 */
void Tensor::init_random(float mean, float std){
    if (data){

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean, std);

        for (int i = 0;i < r;i++){
            for (int j = 0;j < c;j++){
                for (int k = 0;k < d;k++){
                    this->operator()(i, j, k) = distribution(generator);
                }
            }
        }

    } else{
        throw(tensor_not_initialized());
    }
}
/**
     * Constant Initialization
     *
     * Perform the initialization of the tensor to a value v
     *
     * @param r The number of rows
     * @param c The number of columns
     * @param d The depth
     * @param v The initialization value
     */
void Tensor::init(int r, int c, int d, float v){

    if (data != nullptr) throw (unknown_operation());

    this->r = r;
    this->c = c;
    this->d = d;

    data = new float** [r];

    for (int i = 0; i < r; ++i){

        data[i] = new float* [c];

        for (int j = 0; j < c; ++j){

            data[i][j] = new float[d];

            for (int k = 0; k < d; ++k){
                data[i][j][k] = v;
            }
        }
    }
}

/**
 * Tensor Clamp
 *
 * Clamp the tensor such that the lower value becomes low and the higher one become high.
 *
 * @param low Lower value
 * @param high Higher value
 */
void Tensor::clamp(float low, float high){

    if (data == nullptr) throw (tensor_not_initialized());

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                if (data[i][j][k] > high)
                    data[i][j][k] = high;
                else if (data[i][j][k] < low)
                    data[i][j][k] = low;
            }
        }
    }

}

/**
 * Tensor Rescaling
 *
 * Rescale the value of the tensor following this rule:
 *
 * newvalue(i,j,k) = ((data(i,j,k)-min(k))/(max(k)-min(k)))*new_max
 *
 * where max(k) and min(k) are the maximum and minimum value in the k-th channel.
 *
 * new_max is the new maximum value for each channel
 *
 * - if max(k) and min(k) are the same, then the entire k-th channel is set to new_max.
 *
 * @param new_max New maximum vale
 */
void Tensor::rescale(float new_max){

    if (data == nullptr) throw (tensor_not_initialized());

    for (int k = 0; k < d; ++k){

        float max = getMax(k);
        float min = getMin(k);

        if (min != max){
            for (int i = 0; i < r; ++i){
                for (int j = 0; j < c; ++j){

                    data[i][j][k] = ((data[i][j][k] - min) / (max - min)) * new_max;
                }
            }
        } else{
            for (int i = 0; i < r; ++i){
                for (int j = 0; j < c; ++j){

                    data[i][j][k] = new_max;
                }
            }
        }

    }
}

/**
 * Tensor padding
 *
 * Zero pad a tensor in height and width, the new tensor will have the following dimensions:
 *
 * (rows+2*pad_h) x (cols+2*pad_w) x (depth)
 *
 * @param pad_h the height padding
 * @param pad_w the width padding
 * @return the padded tensor
 */
Tensor Tensor::padding(int pad_h, int pad_w)const{

    if (data == nullptr) throw (tensor_not_initialized());

    if (pad_w < 0 || pad_h < 0) throw (unknown_operation());

    Tensor pad(r + 2 * pad_h, c + 2 * pad_w, d, 0.0f);

    for (int i = 0; i < r; ++i){

        for (int j = 0; j < c; ++j){

            for (int k = 0; k < d; ++k){

                pad(i + pad_h, j + pad_w, d) = data[i][j][k];
            }
        }
    }
    return pad;
}

/**
 * Subset a tensor
 *
 * retuns a part of the tensor having the following indices:
 * row_start <= i < row_end
 * col_start <= j < col_end
 * depth_start <= k < depth_end
 *
 * The right extrema is NOT included
 *
 * @param row_start
 * @param row_end
 * @param col_start
 * @param col_end
 * @param depth_start
 * @param depth_end
 * @return the subset of the original tensor
 */
Tensor Tensor::subset(unsigned int row_start, unsigned int row_end, unsigned int col_start, unsigned int col_end, 
                        unsigned int depth_start, unsigned int depth_end) const{

    if (data == nullptr) throw (tensor_not_initialized());

    if (row_start >= r || row_end >= r || row_start < 0 || row_end < 0) throw (index_out_of_bound());
    if (col_start >= c || col_end >= c || col_start < 0 || col_start < 0) throw (index_out_of_bound());
    if (depth_start >= d || depth_end >= d || depth_start < 0 || depth_end < 0) throw (index_out_of_bound());

    if (row_start > row_end) throw (unknown_operation());
    if (col_start > col_end) throw (unknown_operation());
    if (depth_start > depth_end) throw (unknown_operation());


    Tensor sub(row_end - row_start, col_end - col_start, depth_end - depth_start, 0.0f);


    for (int i = row_start; i < row_end; ++i){

        for (int j = col_start; j < col_end; ++j){

            for (int k = depth_start; k < depth_end; ++k){

                sub(i - row_start, j - col_start, k - depth_start) = data[i][j][k];
            }
        }
    }

    return sub;
}

/**
 * Concatenate
 *
 * The function concatenates two tensors along a give axis
 *
 * Example: this is of size 10x5x6 and rhs is of 25x5x6
 *
 * if concat on axis 0 (row) the result will be a new Tensor of size 35x5x6
 *
 * if concat on axis 1 (columns) the operation will fail because the number
 * of rows are different (10 and 25).
 *
 * In order to perform the concatenation is mandatory that all the dimensions
 * different from the axis should be equal, other wise throw concat_wrong_dimension().
 *
 * @param rhs The tensor to concatenate with
 * @param axis The axis along which perform the concatenation
 * @return a new Tensor containing the result of the concatenation
 */
Tensor Tensor::concat(const Tensor& rhs, int axis)const{
    
    if (data == nullptr) throw (tensor_not_initialized());
    if (rhs.data == nullptr) throw (tensor_not_initialized());

    Tensor conc;

    if (axis == 0){
        //rows: vertical
        if (c != rhs.c || d != rhs.d) throw (concat_wrong_dimension());

        conc.init(r + rhs.r, c, d, 0.0f);

        for (int i = 0; i < conc.r; ++i){
            for (int j = 0; j < conc.c; ++j){
                for (int k = 0; k < conc.d; ++k){

                    if (i >= r)
                        conc(i, j, k) = data[i][j][k];
                    else
                        conc(i, j, k) = rhs.data[i - r][j][k];
                }
            }
        }

    } else if (axis == 1){
        //columns: orizontal

        conc.init(r, c + rhs.c, d, 0.0f);

        for (int i = 0; i < conc.r; ++i){
            for (int j = 0; j < conc.c; ++j){
                for (int k = 0; k < conc.d; ++k){

                    if (j >= c)
                        conc(i, j, k) = data[i][j][k];
                    else
                        conc(i, j, k) = rhs.data[i][j - c][k];
                }
            }
        }

    } else if (axis == 2) {
        conc.init(r, c, d + rhs.d, 0.0f);

        for (int i = 0; i < conc.r; ++i){
            for (int j = 0; j < conc.c; ++j){
                for (int k = 0; k < conc.d; ++k){

                    if (k >= d)
                        conc(i, j, k) = data[i][j][k];
                    else
                        conc(i, j, k) = rhs.data[i][j][k - d];
                }
            }
        }
    }

    return conc;
}


/**
 * Convolution
 *
 * This function performs the convolution of the Tensor with a filter.
 *
 * The filter f must have odd dimensions and same depth.
 *
 * Remeber to apply the padding before running the convolution
 *
 * @param f The filter
 * @return a new Tensor containing the result of the convolution
 */
Tensor Tensor::convolve(const Tensor& f)const{

    if(f.data == nullptr) throw(tensor_not_initialized());

    if(data == nullptr) throw(tensor_not_initialized());

    if (f.r != f.c) throw(dimension_mismatch());

    if(f.r % 2 == 0) throw(filter_odd_dimensions());
    if(f.c % 2 == 0) throw(filter_odd_dimensions());

    if(d != f.d) throw(dimension_mismatch());


    int pad = (f.r - 1) / 2;

    Tensor copy = padding(pad, pad);

    Tensor conv(*this);

    for (int i = 0; i < r; ++i){
        for (int j = 0; j < c; ++j){
            for (int k = 0; k < d; ++k){

                float val = 0;

                for (int l = 0; l < f.r; ++l){
                    for (int m = 0; m < f.c; ++m){

                        val += (conv(i + l, j + m, k) * f(l, m, k));
                    }
                }

                conv(i, j, k) = val;
            }
        }
    }
    return conv;
}

/* UTILITY */

/**
 * Rows
 *
 * @return the number of rows in the tensor
 */
int Tensor::rows()const{
    return r;
}

/**
 * Cols
 *
 * @return the number of columns in the tensor
 */
int Tensor::cols()const{
    return c;
}

/**
 * Depth
 *
 * @return the depth of the tensor
 */
int Tensor::depth()const{
    return d;
}

/**
 * Get minimum
 *
 * Compute the minimum value considering a particular index in the third dimension
 *
 * @return the minimum of data( , , k)
 */
float Tensor::getMin(int k)const{
    if (data == nullptr) throw (tensor_not_initialized());

    if(k >= d) throw(index_out_of_bound());

    float min = data[0][0][k];

    for (int i = 0; i < r; ++i){
        for (int j = 0; j < c; ++j){

            if (data[i][j][k] < min)
                min = data[i][j][k];
        }
    }

    return min;
}

/**
 * Get maximum
 *
 * Compute the maximum value considering a particular index in the third dimension
 *
 * @return the maximum of data( , , k)
 */
float Tensor::getMax(int k)const{
    if (data == nullptr) throw (tensor_not_initialized());

    if(k >= d) throw(index_out_of_bound());

    float max = data[0][0][k];

    for (int i = 0; i < r; ++i){
        for (int j = 0; j < c; ++j){

            if (data[i][j][k] > max)
                max = data[i][j][k];
        }
    }

    return max;
}

/**
 * showSize
 *
 * shows the dimensions of the tensor on the standard output.
 *
 * The format is the following:
 * rows" x "colums" x "depth
 *
 */
void Tensor::showSize()const{
    cout << r << " x " << c << " x " << d << endl;
}

/* IOSTREAM */

/**
 * Operator overloading <<
 *
 * Use the overaloading of << to show the content of the tensor.
 *
 * You are free to chose the output format, btw we suggest you to show the tensor by layer.
 *
 * [..., ..., 0]
 * [..., ..., 1]
 * ...
 * [..., ..., k]
 * k: [0 0 0 , 0 0 0, 0 0 0]
*/
ostream& operator<< (ostream& stream, const Tensor& obj){

    //if (obj.data == nullptr) throw (tensor_not_initialized());

    for (int k = 0; k < obj.d; ++k) {
        stream << k << ": [";

        for (int i = 0; i < obj.r; ++i){
            for (int j = 0; j < obj.c; ++j){
                stream << obj(i, j, k);

                if (j != obj.c - 1) stream << " ";
            }

            if (i != obj.r - 1) stream << ",";
        }

        stream << "]" << endl;
    }
    return stream;
}

/**
 * Reading from file
 *
 * Load the content of a tensor from a textual file.
 *
 * The file should have this structure: the first three lines provide the dimensions while
 * the following lines contains the actual data by channel.
 *
 * For example, a tensor of size 4x3x2 will have the following structure:
 * 4
 * 3
 * 2
 * data(0,0,0)
 * data(0,1,0)
 * data(0,2,0)
 * data(1,0,0)
 * data(1,1,0)
 * .
 * .
 * .
 * data(3,1,1)
 * data(3,2,1)
 *
 * if the file is not reachable throw unable_to_read_file()
 *
 * @param filename the filename where the tensor is stored
 */
void Tensor::read_file(string filename){

    ifstream input{ filename };

    input >> r >> c >> d;

    if(data != nullptr) {
        float*** tmp = data;
        data = nullptr;

        if (tmp != nullptr){
            for (int i = 0; i < r; ++i){
                for (int j = 0; j < c; ++j){
                    delete[] tmp[i][j];
                }
                delete[] tmp[i];
            }
            delete[] tmp;
        }
    }

    init(r, c, d, 0.0f);

    for (int i = 0; i < r; ++i){
        for (int j = 0; j < c; ++j){
            for (int k = 0; k < d; ++k){
                input >> data[i][j][k];
            }
        }
    }

}

/**
 * Write the tensor to a file
 *
 * Write the content of a tensor to a textual file.
 *
 * The file should have this structure: the first three lines provide the dimensions while
 * the following lines contains the actual data by channel.
 *
 * For example, a tensor of size 4x3x2 will have the following structure:
 * 4
 * 3
 * 2
 * data(0,0,0)
 * data(0,1,0)
 * data(0,2,0)
 * data(1,0,0)
 * data(1,1,0)
 * .
 * .
 * .
 * data(3,1,1)
 * data(3,2,1)
 *
 * @param filename the filename where the tensor should be stored
 */
void Tensor::write_file(string filename){
    ofstream output{ filename };

    output << r << "\n" << c << "\n" << d << "\n";

    for (int i = 0; i < r; ++i){
        for (int j = 0; j < c; ++j){
            for (int k = 0; k < d; ++k){
                output << data[i][j][k] << "\n";
            }
        }
    }
}

