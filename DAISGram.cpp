#include <iostream>
#include <string>

#include "dais_exc.h"
#include "tensor.h"
#include "libbmp.h"
#include "DAISGram.h"

using namespace std;

DAISGram::DAISGram(){}

DAISGram::~DAISGram(){}




/**
 * Load a bitmap from file
 *
 * @param filename String containing the path of the file
 */
void DAISGram::load_image(string filename){
    BmpImg img = BmpImg();

    img.read(filename.c_str());

    const int h = img.get_height();
    const int w = img.get_width();

    data = Tensor(h, w, 3, 0.0);

    for (int i = 0;i < img.get_height();i++){
        for (int j = 0;j < img.get_width();j++){
            data(i, j, 0) = (float) img.red_at(j, i);
            data(i, j, 1) = (float) img.green_at(j, i);
            data(i, j, 2) = (float) img.blue_at(j, i);
        }
    }
}


/**
 * Save a DAISGram object to a bitmap file.
 *
 * Data is clamped to 0,255 before saving it.
 *
 * @param filename String containing the path where to store the image.
 */
void DAISGram::save_image(string filename){

    data.clamp(0, 255);

    BmpImg img = BmpImg(getCols(), getRows());

    img.init(getCols(), getRows());

    for (int i = 0;i < getRows();i++){
        for (int j = 0;j < getCols();j++){
            img.set_pixel(j, i, (unsigned char) data(i, j, 0), (unsigned char) data(i, j, 1), (unsigned char) data(i, j, 2));
        }
    }

    img.write(filename);

}
/**
         * Get rows
         *
         * @return returns the number of rows in the image
         */
int DAISGram::getRows(){
    return data.rows();
}

/**
 * Get columns
 *
 * @return returns the number of columns in the image
 */
int DAISGram::getCols(){
    return data.cols();
}

/**
 * Get depth
 *
 * @return returns the number of channels in the image
 */
int DAISGram::getDepth(){
    return data.depth();
}

/**
 * Brighten the image
 *
 * It sums the bright variable to all the values in the image.
 *
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 *
 * @param bright the amount of bright to add (if negative the image gets darker)
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::brighten(float bright){
    //Deve aumentare ogni elemento dell'array 3D del valore in ingresso con il cap [0,255]
    DAISGram copy;

    copy.data.init(getRows(), getCols(), getDepth(), 0.0f);

    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){
            for (int k = 0; k < getDepth(); ++k){
                copy.data(i, j, k) = data(i, j, k) + bright;
            }
        }
    }

    copy.data.clamp(0, 255);

    return copy;
}

/**
 * Create a grayscale version of the object
 *
 * A grayscale image is produced by substituting each pixel with its average on all the channel
 *
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::grayscale(){
    //Media del valore dei tre canali per ogni punto e ogni canale avrà gli stessi valori, alla fine avremo tutti i canali uguali
    DAISGram gray;

    gray.data.init(getRows(), getCols(), getDepth(), 0.0f);

    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){

            float m = 0;

            for (int k = 0; k < getDepth(); ++k){
                m += data(i, j, k);
            }

            m /= (float) getDepth();

            for (int k = 0; k < getDepth();++k){
                gray.data(i, j, k) = m;
            }
        }
    }

    return gray;
}

/**
 * Create a Warhol effect on the image
 *
 * This function returns a composition of 4 different images in which the:
 * - top left is the original image
 * - top right is the original image in which the Red and Green channel are swapped
 * - bottom left is the original image in which the Blue and Green channel are swapped
 * - bottom right is the original image in which the Red and Blue channel are swapped
 *
 * The output image is twice the dimensions of the original one.
 *
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::warhol(){
    //top right devo salvare i valori del canale Rosso, Copiare il canale verde su quello rosso e copiare la copia sul verde
    //Dimensione doppia dell'immagine originale

    /*
    ●	In alto a sinistra viene replicata l’immagine originale
    ●	In alto a destra, a partire dall’immagine originale, viene invertito il canale Rosso con il canale Verde
    ●	In basso a sinistra, a partire dall’immagine originale, viene invertito il canale Verde con il canale Blu
    ●	In basso a destra, a partire dall’immagine originale, viene invertito il canale Rosso con il canale Blu
    */
    //immagine normale
    DAISGram top_left;

    top_left.data.init(getRows(), getCols(), getDepth(), 0.0f);
    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){
            for (int k = 0; k < getDepth(); ++k){
                top_left.data(i, j, k) = data(i, j, k);
            }
        }
    }

    //immagine verde
    DAISGram top_right;

    top_right.data.init(getRows(), getCols(), getDepth(), 0.0f);
    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){

            top_right.data(i, j, 0) = data(i, j, 1); //G->R

            top_right.data(i, j, 1) = data(i, j, 0); //R->G

            top_right.data(i, j, 2) = data(i, j, 2); //B
        }
    }

    //immagine rossa
    DAISGram bottom_left;

    bottom_left.data.init(getRows(), getCols(), getDepth(), 0.0f);
    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){

            bottom_left.data(i, j, 2) = data(i, j, 1); //G->B

            bottom_left.data(i, j, 0) = data(i, j, 0); //R

            bottom_left.data(i, j, 1) = data(i, j, 2); //B->G
        }
    }

     //immagine blue
    DAISGram bottom_right;

    bottom_right.data.init(getRows(), getCols(), getDepth(), 0.0f);
    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){

            bottom_right.data(i, j, 2) = data(i, j, 0); //R->B

            bottom_right.data(i, j, 1) = data(i, j, 1); //G

            bottom_right.data(i, j, 0) = data(i, j, 2); //B->R
        }
    }


    Tensor top = top_left.data.concat(top_right.data, 1);
    Tensor bottom = bottom_left.data.concat(bottom_right.data, 1);

    DAISGram complete;
    complete.data = top.concat(bottom, 0);

    return complete;
}

/**
 * Sharpen the image
 *
 * This function makes the image sharper by convolving it with a sharp filter
 *
 * filter[3][3]
 *    0  -1  0
 *    -1  5 -1
 *    0  -1  0
 *
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 *
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::sharpen(){

    DAISGram copy;

    copy.data.init(getRows(), getCols(), getDepth(), 0.0f);

    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){
            for (int k = 0; k < getDepth(); ++k){
                copy.data(i, j, k) = data(i, j, k);
            }
        }
    }

    Tensor filter(3, 3, getDepth(), 0.0f);

    for (int k = 0; k < getDepth(); ++k){

        filter(0, 0, k) = filter(2, 2, k) = filter(0, 2, k) = filter(2, 0, k) = 0;

        filter(0, 1, k) = filter(1, 0, k) = filter(2, 1, k) = filter(1, 2, k) = -1;

        filter(1, 1, k) = 5;
    }

    DAISGram final;

    final.data = copy.data.convolve(filter);

    final.data.clamp(0, 255);

    return final;
}

/**
 * Emboss the image
 *
 * This function makes the image embossed (a light 3D effect) by convolving it with an
 * embossing filter
 *
 * filter[3][3]
 *    -2 -1  0
 *    -1  1  1
 *     0  1  2
 *
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 *
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::emboss(){

    DAISGram copy;
    copy.data.init(getRows(), getCols(), getDepth(), 0.0f);
    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){
            for (int k = 0; k < getDepth(); ++k){
                copy.data(i, j, k) = data(i, j, k);
            }
        }
    }

    Tensor filter(3, 3, getDepth(), 0.0f);

    for (int k = 0; k < getDepth(); ++k){

        filter(0, 0, k) = -2;

        filter(0, 1, k) = filter(1, 0, k) = -1;

        filter(0, 2, k) = filter(2, 0, k) = 0;

        filter(2, 1, k) = filter(1, 2, k) = filter(1, 1, k) = 1;

        filter(2, 2, k) = 2;
    }

    DAISGram final;

    final.data = copy.data.convolve(filter);

    final.data.clamp(0, 255);

    return final;
}

/**
 * Smooth the image
 *
 * This function remove the noise in an image using convolution and an average filter
 * of size h*h:
 *
 * c = 1/(h*h)
 *
 * filter[3][3]
 *    c c c
 *    c c c
 *    c c c
 *
 * @param h the size of the filter
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::smooth(int h){

    float c = (float) 1.0f / (h * h);

    Tensor filter(h, h, getDepth(), c);

    DAISGram final;

    final.data = data.convolve(filter);

    return final;
}

/**
 * Edges of an image
 *
 * This function extract the edges of an image by using the convolution
 * operator and the following filter
 *
 *
 * filter[3][3]
 * -1  -1  -1
 * -1   8  -1
 * -1  -1  -1
 *
 * Remeber to convert the image to grayscale before running the convolution.
 *
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 *
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::edge(){

    DAISGram gray = this->grayscale();

    Tensor filter(3, 3, getDepth(), 0.0f);

    for (int k = 0; k < getDepth(); ++k){

        filter(0, 0, k) = filter(2, 2, k) = filter(0, 2, k) = filter(2, 0, k) = filter(0, 1, k) = filter(1, 0, k) = filter(2, 1, k) = filter(1, 2, k) = -1;
        filter(1, 1, k) = 8;
    }

    DAISGram final;

    final.data = gray.data.convolve(filter);

    final.data.clamp(0, 255);

    return final;
}

/**
 * Blend with anoter image
 *
 * This function generate a new DAISGram which is the composition
 * of the object and another DAISGram object
 *
 * The composition follows this convex combination:
 * results = alpha*this + (1-alpha)*rhs
 *
 * rhs and this obejct MUST have the same dimensions.
 *
 * @param rhs The second image involved in the blending
 * @param alpha The parameter of the convex combination
 * @return returns a new DAISGram containing the blending of the two images.
 */
DAISGram DAISGram::blend(const DAISGram& rhs, float alpha){

    DAISGram copy;

    copy.data.init(getRows(), getCols(), getDepth(), 0.0f);

    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){
            for (int k = 0; k < getDepth(); ++k){
                copy.data(i, j, k) = data(i, j, k) * alpha + (1 - alpha) * rhs.data(i, j, k);
            }
        }
    }

    return copy;
}

/**
 * Green Screen
 *
 * This function substitutes a pixel with the corresponding one in a background image
 * if its colors are in the surrounding (+- threshold) of a given color (rgb).
 *
 * (rgb - threshold) <= pixel <= (rgb + threshold)
 *
 *
 * @param bkg The second image used as background
 * @param rgb[] The color to substitute (rgb[0] = RED, rgb[1]=GREEN, rgb[2]=BLUE)
 * @param threshold[] The threshold to add/remove for each color (threshold[0] = RED, threshold[1]=GREEN, threshold[2]=BLUE)
 * @return returns a new DAISGram containing the result.
 */
DAISGram DAISGram::greenscreen(DAISGram& bkg, int rgb[], float threshold[]){

    //controllo k non maggiore di 3
    //if (data == nullptr) throw(tensor_not_initialized());

    //if (bkg.data == nullptr) throw(tensor_not_initialized());

    if (this->getCols() != bkg.getCols()) throw (dimension_mismatch());

    if (this->getRows() != bkg.getRows()) throw (dimension_mismatch());

    if (this->getDepth() != bkg.getDepth()) throw (dimension_mismatch());

    if (this->getDepth() > 3 || bkg.getDepth() > 3) throw (dimension_mismatch());


    DAISGram copy;

    copy.data = Tensor(data);

    for (int i = 0; i < getRows(); ++i){
        for (int j = 0; j < getCols(); ++j){

            if (data(i, j, 0) >= (rgb[0] - threshold[0]) && data(i, j, 0) <= (rgb[0] + threshold[0])){

                if (data(i, j, 1) >= (rgb[1] - threshold[1]) && data(i, j, 1) <= (rgb[1] + threshold[1])){

                    if (data(i, j, 2) >= (rgb[2] - threshold[2]) && data(i, j, 2) <= (rgb[2] + threshold[2])){

                        for (int k = 0; k < 3; ++k){
                            copy.data(i, j, k) = bkg.data(i, j, k);
                        }


                    }
                }
            }


        }
    }

    return copy;
}
/**
 * Equalize
 *
 * Stretch the distribution of colors of the image in order to use the full range of intesities.
 *
 * See https://it.wikipedia.org/wiki/Equalizzazione_dell%27istogramma
 *
 * @return returns a new DAISGram containing the equalized image.
 */
DAISGram DAISGram::equalize(){

    DAISGram copy;

    //copy = grayscale();

    copy.data = Tensor(data);

    for (int k = 0; k < getDepth(); ++k){

        int count[256];

        for (int i = 0; i < 256; ++i){
            count[i] = 0;
        }

        for (int i = 0; i < getRows(); ++i){
            for (int j = 0; j < getCols(); ++j){

                ++count[(int) copy.data(i, j, k)];
            }
        }

        float equalize[256];

        float cdf_min = count[(int) copy.data.getMin(k)];

        int cdf = 0;

        float den = (copy.getRows() * copy.getCols()) - cdf_min;

        for (int i = (int) copy.data.getMin(k); i < 256; ++i){

            cdf += count[i];

            float delta_cdf = cdf - cdf_min;

            equalize[i] = round( (delta_cdf / den) * 255.0f );
        }

        for (int i = 0; i < copy.getRows(); ++i){
            for (int j = 0; j < copy.getCols(); ++j){

                copy.data(i, j, k) = equalize[ (int) copy.data(i, j, k) ];
            }
        }

    }

    return copy;
}

/**
 * Generate Random Image
 *
 * Generate a random image from nois
 *
 * @param h height of the image
 * @param w width of the image
 * @param d number of channels
 * @return returns a new DAISGram containing the generated image.
 */
void DAISGram::generate_random(int h, int w, int d){
    data = Tensor(h, w, d, 0.0);
    data.init_random(128, 50);
    data.rescale(255);
}

