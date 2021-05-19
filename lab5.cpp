
#include <opencv2/opencv.hpp>
#include "PanoramicImage.h"
#include <stdlib.h>

using namespace cv;
using namespace std;

const double default_half_fov = 33;
const string default_img_directory_path = "../data/";

const double lowe_ratio = 0.4;


//>> lab5 dir_path half_fov

int main(int argc, char* argv[]) {
	
    //command line parsing
    String directory_path = "";
    double half_fov = default_half_fov;
    if (argc > 1) {
        String tmp = argv[1];
        half_fov = atof(argv[2]);
        directory_path = directory_path + tmp;
        cout << "reading from folder: " << directory_path << endl;
    }
    else {
        directory_path = directory_path + default_img_directory_path;
        cout << "reading from folder: " << directory_path << endl;
    }


	PanoramicImage pan = PanoramicImage();

    //loading and projectig images
    cout << "loading images.. " << endl;
	pan.load_and_project_imgs(directory_path, half_fov);

    cout << "finding features..." << endl;

    //find features
    pan.find_features();

    //find translations
    pan.find_translations(lowe_ratio);

    //compute panorama
    Mat output_image = pan.compute_panorama();

    imshow("Panorama", output_image);

    waitKey(0);
	return 0;
}