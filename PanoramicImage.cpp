#include "PanoramicImage.h"


PanoramicImage::PanoramicImage(){

}

void PanoramicImage::load_and_project_imgs(String img_folder_path, double half_fov) {
	vector<String> img_filenames;
	utils::fs::glob(img_folder_path, "*.bmp", img_filenames);
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
		/*
		Mat output;
		drawKeypoints(imgs[i], kp_vector[i], output);
		imshow("dfdsfsd", output);
		waitKey(0);
		*/
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

		float dx_accumulator = 0;
		float dy_accumulator = 0;

		for (int j = 0; j < matches.size()-1; j++) {
			if (matches[j][0].distance < ratio * matches[j][1].distance) {
				DMatch good_match = matches[j][0];
				good_matches.push_back(good_match); // add this match to the good matches
				Point2f p1 = kp_vector[i][good_match.queryIdx].pt;
				Point2f p2 = kp_vector[i + 1][good_match.trainIdx].pt;
				src_pts.push_back(p1); //add the first image point from the current good match 
				dst_pts.push_back(p2); //add the second image point from the current good match
				//calculate points distance		
				dx_accumulator += p1.x - p2.x;
				dy_accumulator += p1.y - p2.y;
			}
		}
		//calculate avarage distance
		float dx_avg = dx_accumulator / size(good_matches);
		float dy_avg = dy_accumulator / size(good_matches);

		//push em
		dx_avgs.push_back(dx_avg);
		dy_avgs.push_back(dy_avg);

		std::cout << "dx_avg = " << dx_avg << endl;
		std::cout << "dy_avg = " << dy_avg << endl;

		good_matches_vec.push_back(good_matches); //add this vector of good matches to the vector of vectors of good matches

		//Mat out;
		//cv::drawMatches(imgs[i], kp_vector[i], imgs[i+1], kp_vector[i + 1], good_matches, out);
		//cv::imshow("Matches", out);
		//cv::waitKey(0);
	}
}





Mat PanoramicImage::compute_panorama() {
	int out_rows = imgs[0].rows;
	int out_cols = imgs[0].cols;
	
	for (int i = 0; i < dx_avgs.size(); i++) {
		out_cols += imgs[i + 1].cols - cvRound(dx_avgs[i]);
	}

	cout << "Single image is " << imgs[0].rows << " x " << imgs[0].cols << endl;
	cout << "Full panorama is " << out_rows << " x " << out_cols << endl;


	Mat out = Mat::zeros(Size(out_cols, out_rows), imgs[0].type());

	///*
	Mat m0 = out.colRange(0, imgs[0].cols);
	imgs[0].copyTo(m0);

	int cols_acc = imgs[0].cols;
	for (int i = 0; i < dx_avgs.size(); i++) {
		Mat tmp_img = imgs[i + 1];
		cols_acc += tmp_img.cols - cvRound(dx_avgs[i]);
		Mat mi = out.colRange(cols_acc-tmp_img.cols, cols_acc);
		//equalizeHist(tmp_img, tmp_img); //made it worse
		tmp_img.copyTo(mi);
	}
	//*/

	/*
	//backwards
	int last_elem = dx_avgs.size() - 1;
	Mat mlast = out.colRange(out_cols- imgs[last_elem].cols, out_cols);
	imgs[0].copyTo(mlast);

	int cols_acc = out_cols - imgs[last_elem].cols;
	for (int i = last_elem; i > 0; i--) {
		cout << "Im here " << i << endl;
		Mat tmp_img = imgs[i + 1];
		cols_acc += tmp_img.cols - dx_avgs[i];
		Mat mi = out.colRange(cols_acc - tmp_img.cols, cols_acc);
		//equalizeHist(tmp_img, tmp_img); //made it worse
		Mat tmp_mask = Mat::ones(out.size(), out.type());
		tmp_img.copyTo(mi, tmp_mask.colRange(cols_acc - tmp_img.cols, cols_acc + 1));
	}
	*/



	return out;
}




