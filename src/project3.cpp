#include <iostream>
#include "cv.hpp"
#include <fstream>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

void Detect_lane(Mat&, bool*);
float LineMerg(Mat& frame, vector<Vec2f>& lines, int line_size, int, int);
void MyCarStop(Mat frame_clone, Mat& avg);
bool Detect_movFstop(Mat cur_frame, Net& net, vector<String>& classNamesVec);
String Detect_NearbyObj(Mat&, Net& net, vector<String>& classNamesVec);
void DisplayAlert(Mat& frame, bool* detect_obj, VideoCapture&);

bool myCarStop = false;
int frame_cnt[4] = { 0,0,0,0 };
int detect_wndWidth[2] = { 0,0 }, wndW_idx = 0;

int main() {
    Mat frame, prev_frame, frame_clone, avg;
    float delay;
    int fps, prev_frame_cnt = 0, cur_frame_cnt = 0, cnt = 2;
    String modelConfiguration = "yolov2-tiny.cfg";
    String modelBinary = "yolov2-tiny.weights";
    vector<String> classNamesVec;
    bool detect_obj[4] = { 0,0,0,0 }; //0. lane departure, 1. stop->moving, 2.nearbyCar 3.nearbyPerson

    //load deep learning model_Yolo
    Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    ifstream classNamesFile("coco.names");

    if (classNamesFile.is_open()) {
        string className = "";
        while (std::getline(classNamesFile, className)) classNamesVec.push_back(className);
    }

    //read mp4
    VideoCapture cap;
    if (cap.open("../Computer_Vision_dataset2/project3_1.mp4") == 0) {
        cout << "No such file" << endl;
        return -1;
    }

    fps = cap.get(CV_CAP_PROP_FPS);
    delay = 1000 / fps;
    cap >> avg;

    while (1) {
        String objName;

        //현재 frame
        cap >> frame; //720x480
        frame_clone = frame.clone();

        if (frame.empty()) break;

        cur_frame_cnt += 1;

        //detect lane depature of our car
        Detect_lane(frame, detect_obj); //use lane detection(houghlines)

        //background Subtraction기법 사용 (avg=background)
        add(frame_clone / cnt, avg * (cnt - 1) / cnt, avg);
        cnt++;

        //속도 향상을 위해 매 15frame마다 yolo사용
        if (cur_frame_cnt - prev_frame_cnt >= 15) {
            cnt = 2;
            prev_frame_cnt = cur_frame_cnt;

            //detect my car is stationary
            MyCarStop(frame_clone, avg);

            //내 차가 정차하고 있을 때, 앞차가 정차 후 움직이는 것 감지
            if (myCarStop == true)
                detect_obj[1] = Detect_movFstop(frame, net, classNamesVec);
            //else면 windowSize저장 배열 초기화 
            else
                detect_wndWidth[0] = detect_wndWidth[1] = 0;

            //detect person and car are detected nearby while driving
            objName = Detect_NearbyObj(frame, net, classNamesVec);

            if (objName == "car") {
                detect_obj[2] = true;
                frame_cnt[2] = 0;
            }
            else if (objName == "person") {
                detect_obj[3] = true;
                frame_cnt[3] = 0;
            }
        }

        //display alert message on the video
        DisplayAlert(frame, detect_obj, cap);

        imshow("Project3", frame);
        waitKey(20);
    }

    return 0;
}

void Detect_lane(Mat& frame, bool detect_obj[]) {
    Mat frame_gray, frame_roi;
    int frame_w = frame.cols;
    int frame_h = frame.rows;
    Rect rect(frame_w / 3, frame_h / 2, frame_w / 3, frame_h / 2);
    vector<Vec2f> lines;
    int err = 40;
    float line_cx;

    //cvtColor로 gray scale 변환
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    frame_roi = frame_gray(rect);

    //blur func 사용해서 noise 제거
    GaussianBlur(frame_roi, frame_roi, Size(5, 5), 0, 0, BORDER_DEFAULT);


    //canny edge detect 적용(TL=10, TH=60)
    Canny(frame_roi, frame_roi, 10, 60, 3);

    //line filtering (houghlines transform) #input #output #rho resolution #theta resolution #threshold #srn #s? #min angle #max angle
    //threshold : 해상도 안의 겹치는 수가 몇개 이상이여야 직선으로 인정할 것인지
    HoughLines(frame_roi, lines, 2, CV_PI / 180, 150, 0, 0, 160 * CV_PI / 180, 190 * CV_PI / 180);

    //line merging (Take average of tho & theta of tiltered line)
    line_cx = LineMerg(frame, lines, lines.size(), frame_w / 3, frame_h / 2);


    //detect lane
    if ((line_cx >= frame.cols / 2 - err) && (line_cx <= frame.cols / 2 + err)) {
        detect_obj[0] = true;
        frame_cnt[0] = 0;
    }
}

float LineMerg(Mat& frame, vector<Vec2f>& lines, int line_size, int plus_x, int plus_y) {
    float rho, theta, rho_sum = 0, rho_avg = 0, theta_sum = 0, theta_avg = 0;
    float a, b, x0 = 0, y0 = 0, line_cx, line_cy;
    Point p1, p2;

    for (int i = 0; i < line_size; i++) {
        rho = lines[i][0];
        theta = lines[i][1];

        rho_sum += rho;
        theta_sum += theta;
    }
    //많은 라인 중 평균을 취해서 대표 라인 정하기
    rho_avg = rho_sum / line_size;
    theta_avg = theta_sum / line_size;

    //info about Line
    a = cos(theta_avg);
    b = sin(theta_avg);

    x0 = a * rho_avg;
    y0 = b * rho_avg;
    line_cx = x0 + plus_x;
    //p1 = Point(cvRound(x0 + 1000 * (-b)) + 240, cvRound(y0 + 1000 * a) + 240);
    //p2 = Point(cvRound(x0 - 1000 * (-b)) + 240, cvRound(y0 - 1000 * a) + 240);

    //line(frame, p1, p2, Scalar(0, 0, 255), 3, 8);
    return line_cx;
}

void MyCarStop(Mat fm_clone, Mat& avg) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat clone_gray = fm_clone.clone();
    int frame_w = fm_clone.cols;
    int frame_h = fm_clone.rows;
    int err = 50;
    Rect rect(0, 0, frame_w, frame_h / 2);

    clone_gray = clone_gray(rect);
    avg = avg(rect);
    //imshow("avg", avg);
    cvtColor(clone_gray, clone_gray, CV_BGR2GRAY);
    cvtColor(avg, avg, CV_BGR2GRAY);
    //imshow("gray_avg", avg);
    absdiff(clone_gray, avg, avg);
    threshold(avg, avg, 100, 255, CV_THRESH_BINARY); //binary img
    //imshow("diff", avg);
    findContours(avg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


    //Our car is stopping.
    if (contours.size() <= 80)
        myCarStop = true;
    else
        myCarStop = false;

    avg = fm_clone.clone();
}

bool Detect_movFstop(Mat frame, Net& net, vector<String>& classNamesVec) {
    int frame_w = frame.cols;
    int frame_h = frame.rows;
    int err = 50;
    Rect rect(frame_w / 3 - err, frame_h / 2 - err * 2, frame_w / 3 + err * 2, frame_h / 2 + err * 2);

    //내 차 앞부분만으로 detect하기 위해 이미지 자르기
    frame = frame(rect);
    if (frame.channels() == 4) cvtColor(frame, frame, COLOR_BGRA2BGR);

    Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
    net.setInput(inputBlob, "data");
    Mat detectionMat = net.forward("detection_out");

    float confidenceThreshold = 0.24;

    for (int i = 0; i < detectionMat.rows; i++) {
        const int probability_index = 5;
        const int probability_size = detectionMat.cols - probability_index;
        float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);
        size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

        if (confidence > confidenceThreshold) {
            float x_center = detectionMat.at<float>(i, 0) * frame.cols;
            float y_center = detectionMat.at<float>(i, 1) * frame.rows;
            float width = detectionMat.at<float>(i, 2) * frame.cols;
            float height = detectionMat.at<float>(i, 3) * frame.rows;
            String objName;

            if ((frame.cols / 3 <= x_center && x_center <= frame.cols * 2 / 3) && (frame.rows / 3 <= y_center && y_center <= frame.rows * 2 / 3)) {
                detect_wndWidth[wndW_idx] = width;
                wndW_idx += 1;
                wndW_idx = wndW_idx % 2;
            }
        }


    }
    //앞 차가 정차 후, 움직임을 감지 start moving
    if (detect_wndWidth[0] != 0 && detect_wndWidth[1] != 0)
        if (abs(detect_wndWidth[0] - detect_wndWidth[1]) > 70)
            return true;

    return false;
}

String Detect_NearbyObj(Mat& frame, Net& net, vector<String>& classNamesVec) {
    if (frame.channels() == 4) cvtColor(frame, frame, COLOR_BGRA2BGR);

    Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
    net.setInput(inputBlob, "data");
    Mat detectionMat = net.forward("detection_out");

    float confidenceThreshold = 0.24;

    for (int i = 0; i < detectionMat.rows; i++) {
        const int probability_index = 5;
        const int probability_size = detectionMat.cols - probability_index;
        float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);
        size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

        if (confidence > confidenceThreshold) {
            float x_center = detectionMat.at<float>(i, 0) * frame.cols;
            float y_center = detectionMat.at<float>(i, 1) * frame.rows;
            float width = detectionMat.at<float>(i, 2) * frame.cols;
            float height = detectionMat.at<float>(i, 3) * frame.rows;
            String objName;

            Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
            Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
            Rect object(p1, p2);

            String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);

            objName = className.c_str();
            if (objName == "car") {
                Scalar object_roi_color(0, 0, 255);
                rectangle(frame, object, object_roi_color);
            }
            else if (objName == "person") {
                Scalar object_roi_color(0, 255, 0);
                rectangle(frame, object, object_roi_color);
            }

            //detect nearbyObject
            if (width >= frame.cols / 5 || height >= frame.rows / 5)
                return objName;
        }
    }
    return "fail";
}

void DisplayAlert(Mat& frame, bool* detect_obj, VideoCapture& cap) {
    String label[4] = {};
    label[0] = format("Lane departure");
    label[1] = format("Start moving");
    label[2] = format("Car Detected nearby!");
    label[3] = format("Person Detected nearby!");

    for (int i = 0; i < 4; i++) {
        if (detect_obj[i] == true) {
            frame_cnt[i]++;

            if (frame_cnt[i] <= 40) {
                putText(frame, label[i], Point(frame.cols / 5, (i + 1) * (frame.rows / 8)), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(63 * i / 2, 63 * i, 150), 4);
            }
            else {
                frame_cnt[i] = 0;
                detect_obj[i] = false;
            }
        }
    }
}
