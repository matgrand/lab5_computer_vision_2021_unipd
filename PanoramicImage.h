#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "panoramic_utils.h"

#define SLOW_MODE false

using namespace cv;
using namespace std;

class PanoramicImage {
public:

	// constructor
	PanoramicImage();

	// // methods
	//load a set of images
	void load_and_project_imgs(String img_folder_path, double half_fov);

	//extract SIFT features
	void find_features();

	//for each couple 1-compute match, 2-Refine match, 3-find translations
	void find_translations(double ratio);

	//compute final panorama
	Mat compute_panorama();

	// //variables

	vector<Mat> imgs; //vector of loaded images, in order
	vector<vector<KeyPoint>> kp_vector;
	vector<Mat> sift_masks;
	vector<Mat> sift_descriptors;
	vector<vector<vector<DMatch>>> matches_vec;
	vector<vector<DMatch>> good_matches_vec;

	vector<int> dx_avgs;
	vector<int> dy_avgs;

};

