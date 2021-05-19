#include "PanoramicImage.h"
#include <numeric>


float mode(vector<int> v);

PanoramicImage::PanoramicImage(){

}

void PanoramicImage::load_and_project_imgs(String img_folder_path, double half_fov) {
	vector<String> img_filenames;
	utils::fs::glob(img_folder_path, "*.bmp", img_filenames);
	utils::fs::glob(img_folder_path, "*.png", img_filenames);
	//add the projected images to imgs
	for (int i = 0; i < size(img_filenames); i++) {
		Mat tmp_img = imread(img_filenames[i]);
		imgs.push_back(PanoramicUtils::cylindricalProj(tmp_img, half_fov));
		
		//initializing everything
		vector<KeyPoint> tmp_kp = { KeyPoint() };
		vector<DMatch> tmp_match = { DMatch(), DMatch() };
		vector<vector<DMatch>> tmp_vect_match = { tmp_match };
		Mat tmp_mat = imgs[i];
		kp_vector.push_back(tmp_kp);
		sift_descriptors.push_back(tmp_mat);
		sift_masks.push_back(tmp_mat);
		matches_vec.push_back(tmp_vect_match);
	}
	matches_vec.pop_back(); //has 1 element less
}

//find SIFT features
void PanoramicImage::find_features() {
	cv::Ptr<SIFT> siftPtr = SIFT::create();
	for (int i = 0; i < size(imgs); i++) {
		siftPtr->detectAndCompute(imgs[i],sift_masks[i], kp_vector[i], sift_descriptors[i], false);
		if (SLOW_MODE) {
			Mat output;
			drawKeypoints(imgs[i], kp_vector[i], output);
			imshow("dfdsfsd", output);
			waitKey(0);
		}
	}
}


//find translations
void PanoramicImage::find_translations(double ratio) {
	BFMatcher bf = BFMatcher(NORM_L2, false);

	dx_avgs = {};
	dy_avgs = {};

	for (int i = 0; i < size(imgs)-1; i++) {
	
		bf.knnMatch(sift_descriptors[i], sift_descriptors[i+1], matches_vec[i], 2); //try sift_masks[i] instead of noArray

		//keep only good matches, nearest neighbor distance ratio matching (not specified like this in handout)
		vector<vector<DMatch>> matches = matches_vec[i];
		vector<DMatch> good_matches;
		vector<Point2f> src_pts;
		vector<Point2f> dst_pts;

		vector<float> dx_vec;
		vector<float> dy_vec;

		for (int j = 0; j < matches.size()-1; j++) {
			if (matches[j][0].distance < ratio * matches[j][1].distance) {
				DMatch good_match = matches[j][0];
				good_matches.push_back(good_match); // add this match to the good matches
				Point2f p1 = kp_vector[i][good_match.queryIdx].pt;
				Point2f p2 = kp_vector[i + 1][good_match.trainIdx].pt;
				src_pts.push_back(p1); //add the first image point from the current good match 
				dst_pts.push_back(p2); //add the second image point from the current good match
				
				// points distance		
				dx_vec.push_back(abs(p1.x - p2.x));
				dy_vec.push_back(p1.y - p2.y);
			}
		}
		
		//calculate mean distance
		float dx_avg = accumulate(dx_vec.begin(), dx_vec.end(), 0.0) / size(good_matches);
		float dy_avg = accumulate(dy_vec.begin(), dy_vec.end(), 0.0) / size(good_matches);
		
		//calculate mode
		//int dx_avg = mode(dx_vec);
		//int dy_avg = mode(dy_vec);

		//push em
		dx_avgs.push_back(cvRound(dx_avg));
		dy_avgs.push_back(cvRound(dy_avg));

		std::cout << "dx_avg = " << dx_avg << endl;
		std::cout << "dy_avg = " << dy_avg << endl;

		good_matches_vec.push_back(good_matches); //add this vector of good matches to the vector of vectors of good matches

		if (SLOW_MODE) {
			Mat out;
			cv::drawMatches(imgs[i], kp_vector[i], imgs[i+1], kp_vector[i + 1], good_matches, out);
			cv::imshow("Matches", out);
			cv::waitKey(0);
		}
	}
}





Mat PanoramicImage::compute_panorama() {
	int out_rows = imgs[0].rows;
	int out_cols = 0;

	
	//get x displacement and crop each image
	int bef_cut = 0;
	int aft_cut = imgs[0].cols - (dx_avgs[0] - dx_avgs[0] / 2);
	vector<int> x_disp = {0};
	vector<Mat> cropped_imgs = {imgs[0].colRange(0,aft_cut)};
	for (int i = 1; i < dx_avgs.size(); i++) {
		
		bef_cut = dx_avgs[i-1] - dx_avgs[i-1] / 2;
		aft_cut = dx_avgs[i] / 2;
		x_disp.push_back(cropped_imgs[i-1].cols + x_disp[i-1]);
		cropped_imgs.push_back(imgs[i].colRange(bef_cut, imgs[i].cols - aft_cut));
	}
	x_disp.push_back(cropped_imgs[cropped_imgs.size() - 1].cols + x_disp[x_disp.size() - 1]);
	cropped_imgs.push_back(imgs[imgs.size()-1].colRange(dx_avgs[dx_avgs.size()-1] - aft_cut, imgs[imgs.size() - 1].cols));

	//show images
	if (SLOW_MODE) {
		for (int i = 0; i < cropped_imgs.size(); i++) {
			imshow("sdfdsf", cropped_imgs[i]);
			cout << "x_disp = " << x_disp[i] << "    witdth = " << cropped_imgs[i].cols << endl;
			waitKey(0); //slow mode
		}
	}
	//calculate #cols
	cout << "# of cropped images = " << cropped_imgs.size() << endl;
	out_cols = 0;
	for (int i = 0; i < cropped_imgs.size(); i++) {
		out_cols += cropped_imgs[i].cols;
	}

	//calculate # rows
	vector<int> dy_prog_sum = {0};
	int dy_acc = 0;
	for (int i = 0; i < dy_avgs.size(); i++) {
		dy_acc += dy_avgs[i];
		dy_prog_sum.push_back(dy_acc);
	}
	//find max and min dy
	int max_dy = *max_element(dy_prog_sum.begin(), dy_prog_sum.end());
	int min_dy = *min_element(dy_prog_sum.begin(), dy_prog_sum.end());
	out_rows = imgs[0].rows + max_dy - min_dy;

	//get y displacement for each image
	vector<int> y_disp;
	for (int i = 0; i < dy_prog_sum.size(); i++) {
		y_disp.push_back(dy_prog_sum[i] - min_dy);
		cout << "y_disp = " << y_disp[i] << endl;
	}

	cout << "Single image is " << imgs[0].rows << " x " << imgs[0].cols << endl;
	cout << "Full panorama is " << out_rows << " x " << out_cols << endl;

	Mat out = Mat::zeros(Size(out_cols, out_rows), cropped_imgs[0].type());

	//compose the image
	for (int i = 0; i < cropped_imgs.size(); i++) {
		Mat mi = out(Rect(x_disp[i], y_disp[i], cropped_imgs[i].cols, cropped_imgs[i].rows));
		cropped_imgs[i].copyTo(mi);
	}

	return out;
}



float mode(std::vector<int> vec) {
	int size = vec.size();
	int max = *max_element(std::begin(vec), std::end(vec));

	std::vector<int> histogram(max+1, 0);
	for (int i = 0; i < vec.size(); i++)
		histogram[vec[i]] += 1;
	
	return std::max_element(histogram.begin(), histogram.end()) - histogram.begin();
}
