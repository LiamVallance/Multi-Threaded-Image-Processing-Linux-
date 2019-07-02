#include <iostream>
#include <vector>
//Thread building blocks library
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <math.h>
#include <tbb/tick_count.h>
//Free Image library
#include <FreeImagePlus.h>

using namespace std;
using namespace tbb;

const float PI = 3.142f;
float sigma = 10.5;
const int KERNEL_SIZE = 15;
int halfKernel = KERNEL_SIZE/2;
int step_size = 250;

template <typename T>
T sqr(const T& a) {

    return a * a;
}


double kernel[KERNEL_SIZE][KERNEL_SIZE] = {0};
/* {
        0.007202,	0.007502,	0.007744,	0.007922,	0.00803,    0.008066,	0.00803,    0.007922,	0.007744,	0.007502,	0.007202,
        0.007502,	0.007815,	0.008066,	0.008251,	0.008364,	0.008402,	0.008364,	0.008251,	0.008066,	0.007815,	0.007502,
        0.007744,	0.008066,	0.008326,	0.008517,	0.008634,	0.008673,	0.008634,	0.008517,	0.008326,	0.008066,	0.007744,
        0.007922,	0.008251,	0.008517,	0.008713,	0.008832,	0.008872,	0.008832,	0.008713,	0.008517,	0.008251,	0.007922,
        0.00803,    0.008364,	0.008634,	0.008832,	0.008953,	0.008993,	0.008953,	0.008832,	0.008634,	0.008364,	0.00803,
        0.008066,	0.008402,	0.008673,	0.008872,	0.008993,	0.009034,	0.008993,	0.008872,	0.008673,	0.008402,	0.008066,
        0.00803,    0.008364,	0.008634,	0.008832,	0.008953,	0.008993,	0.008953,	0.008832,	0.008634,	0.008364,	0.00803,
        0.007922,	0.008251,	0.008517,	0.008713,	0.008832,	0.008872,	0.008832,	0.008713,	0.008517,	0.008251,	0.007922,
        0.007744,	0.008066,	0.008326,	0.008517,	0.008634,	0.008673,	0.008634,	0.008517,	0.008326,	0.008066,	0.007744,
        0.007502,	0.007815,	0.008066,	0.008251,	0.008364,	0.008402,	0.008364,	0.008251,	0.008066,	0.007815,	0.007502,
        0.007202,	0.007502,	0.007744,	0.007922,	0.00803,    0.008066,	0.00803,    0.007922,	0.007744,	0.007502,	0.007202,
};*/

void gaussianSequential(int height, int width, const float *inputBuffer, float *outputBuffer) {

    for (int y =  halfKernel; y < height - halfKernel; y++) {
        for (int x = halfKernel; x < width - halfKernel; x++) {
            for (int i = 0 ;i < KERNEL_SIZE; i++) {
                for (int j =0; j < KERNEL_SIZE; j++){
                    outputBuffer[y * width + x] += inputBuffer[(y - (i - halfKernel)) * width + (x - (j - halfKernel))] * kernel[i][j];
                }
            }
        }
    }
}

void gaussianParallel(int height, int width, const float *inputBuffer, float *outputBuffer, int step_size) {
    parallel_for(blocked_range2d<uint64_t>(halfKernel,height - halfKernel,step_size,halfKernel,width - halfKernel,step_size),[&](const blocked_range2d<uint64_t>&r) {

        auto y1 = r.rows().begin();
        auto y2 = r.rows().end();
        auto x1 = r.cols().begin();
        auto x2 = r.cols().end();

        for (int y = y1; y < y2; y++) {
            for (int x = x1; x < x2; x++) {
                for (int i= 0 ;i<KERNEL_SIZE; i++) {
                    for (int j=0; j<KERNEL_SIZE; j++){

                        outputBuffer[y * width + x] += inputBuffer[(y - (i - halfKernel)) * width + (x - (j - halfKernel))] * kernel[i][j];
                    }
                }
            }
        }

    });
}

int main()
{
    int nt = task_scheduler_init::default_num_threads();
    task_scheduler_init T(nt);

    //Part 1 (Greyscale Gaussian blur): -----------DO NOT REMOVE THIS COMMENT----------------------------//
    fipImage greyscale;
    greyscale.load("../Images/render_1.png");
    greyscale.convertToFloat();


    auto grey_width = greyscale.getWidth();
    auto grey_height = greyscale.getHeight();
    const float* const inputBuffer = (float*)greyscale.accessPixels();



    fipImage output_greyscale;
    output_greyscale = fipImage(FIT_FLOAT, grey_width, grey_height, 32);
    float *outputBuffer = (float *) output_greyscale.accessPixels();
    cout << "Parallel kernel size " << KERNEL_SIZE << ": " << endl;
    for (int i = 0; i < 5; i++) {
        tick_count tStart = tick_count::now();
        double sum = 0.0;
        for (int x = -halfKernel; x <= halfKernel; x++) {
            for (int y = -halfKernel; y <= halfKernel; y++) {
                kernel[x + halfKernel][y + halfKernel] =
                        1.0f / (2.0f * PI * sqr(sigma)) * exp(-((sqr(x) + sqr(y)) / (2.0f * sqr(sigma))));
                sum += kernel[x + halfKernel][y + halfKernel];
            }
        }
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                kernel[i][j] /= sum;
            }
        }
        //gaussianSequential(grey_height, grey_width, inputBuffer, outputBuffer);
        gaussianParallel(grey_height, grey_width, inputBuffer, outputBuffer, step_size);
        tick_count tEnd = tick_count::now();
        cout << "Run Time " << i + 1 << ": " << (tEnd - tStart).seconds() << " seconds\n";
    }
        cout << endl;

    output_greyscale.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    output_greyscale.convertTo24Bits();
    output_greyscale.save("../Images/grey_blurred.png");

    //Part 2 (Colour image processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//

    // Setup Input image array
    fipImage inputImage;
    inputImage.load("../Images/render_1.png");
    // Setup Input image array 2
    fipImage inputImage2;
    inputImage2.load("../Images/render_2.png");

    unsigned int width = inputImage.getWidth();
    unsigned int height = inputImage.getHeight();


    // Setup Output image array
    fipImage outputImage;
    outputImage = fipImage(FIT_BITMAP, width, height, 24);

    //2D Vector to hold the RGB colour data of an image
    vector<vector<RGBQUAD>> rgbValues;
    rgbValues.resize(height, vector<RGBQUAD>(width));

    vector<vector<RGBQUAD>> rgbValuesDiff;
    rgbValuesDiff.resize(height, vector<RGBQUAD>(width));

    //RGBQUAD rgb;  //FreeImage structure to hold RGB values of a single pixel
    //RGBQUAD rgb2;  //FreeImage structure to hold RGB values of a single pixel

    int white = 0, black = 0;
    float parallel_total = 0, parallel_percentage = 0;
    const int THRESHOLD = 20;

    parallel_for(blocked_range2d<uint64_t>(0,height,0,width),[&](const blocked_range2d<uint64_t>&r) {

        auto y1 = r.rows().begin();
        auto y2 = r.rows().end();
        auto x1 = r.cols().begin();
        auto x2 = r.cols().end();

        int final;

        RGBQUAD rgb;  //FreeImage structure to hold RGB values of a single pixel
        RGBQUAD rgb2;  //FreeImage structure to hold RGB values of a single pixel

        for (int y=y1; y!=y2; y++) {
            for (int x=x1; x!=x2; x++) {


                inputImage.getPixelColor(x, y, &rgb); //Extract pixel(x,y) colour data and place it in rgb
                inputImage2.getPixelColor(x, y, &rgb2); //Extract pixel(x,y) colour data and place it in rgb

                rgbValuesDiff[y][x].rgbRed = abs(rgb.rgbRed - rgb2.rgbRed);
                rgbValuesDiff[y][x].rgbGreen = abs(rgb.rgbGreen - rgb2.rgbGreen);
                rgbValuesDiff[y][x].rgbBlue = abs(rgb.rgbBlue - rgb2.rgbBlue);

                final = rgbValuesDiff[y][x].rgbRed + rgbValuesDiff[y][x].rgbGreen + rgbValuesDiff[y][x].rgbBlue;

                if (final >= THRESHOLD) {
                    rgbValuesDiff[y][x].rgbRed = 255;
                    rgbValuesDiff[y][x].rgbGreen = 255;
                    rgbValuesDiff[y][x].rgbBlue = 255;
                } else {
                    rgbValuesDiff[y][x].rgbRed = 0;
                    rgbValuesDiff[y][x].rgbGreen = 0;
                    rgbValuesDiff[y][x].rgbBlue = 0;
                }
                outputImage.setPixelColor(x, y, &rgbValuesDiff[y][x]);
            }
        }

    });

    float x = parallel_reduce(
            blocked_range2d<uint64_t>(0,height,0,width),
            0,
            [&](const blocked_range2d<uint64_t>&r, float initValue) {

        auto y1 = r.rows().begin();
        auto y2 = r.rows().end();
        auto x1 = r.cols().begin();
        auto x2 = r.cols().end();
        float colour;

        for (int y=y1; y!=y2; y++) {
            for (int x=x1; x!=x2; x++) {
                colour = rgbValuesDiff[y][x].rgbRed + rgbValuesDiff[y][x].rgbGreen + rgbValuesDiff[y][x].rgbBlue;
                if (colour >= 255)
                {
                    initValue++;
                }
            }
        }
        return initValue; },
        [&](float x, float y)->float {
            return x + y;
        }
    );

    parallel_total = outputImage.getWidth() * outputImage.getHeight();
    parallel_percentage = (x/parallel_total)*100;
    //Save the processed image
    outputImage.save("../Images/RGB_processed.png");
    cout << "reduce white = " << x << endl;
    cout << "reduce percentage = " << parallel_percentage << "%" << endl;

    return 0;
}