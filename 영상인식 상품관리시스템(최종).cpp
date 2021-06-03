#include<iostream>// c++관련 헤더파일
#include<opencv2/opencv.hpp>// opencv 헤더파일
#include<thread>// 스레드 관련 헤더파일

using namespace std;// 네임스페이스 std 정의
using namespace cv;// 네임스페이스 cv 정의
using namespace cv::dnn;// 네임스페이스 dnn정의

void real_time_video();// 실시간 카메라 영상을 출력하는 함수
void finger_counting();// 실시간 손가락 개수 세는 함수
void predict_age_gender();// 나이, 성별 예측 함수

// 남성(0-2): 우유, 남성(4-6): 우유, 남성(8-12): 사이다, 
// 남성(15-20): 사이다, 남성(25-32): 막걸리, 남성(48-53): 홍삼차, 남성(60-100): 홍삼차
// 여성(0-2): 우유, 여성(4-6): 우유, 여성(8-12): 사이다, 
// 여성(15-20): 토레타, 여성(25-32): 토레타, 여성(48-53): 토레타, 여성(60-100): 홍삼차

Mat getHandMask1(void* src, int minCr, int maxCr, int minCb, int maxCb);// 손을 검출하기위한 함수
Point getHandCenter(const Mat& mask, double& radius);// 손바닥의 중심점,손바닥 영역의 반지름을 구하기 위한 함수
int getFingerCount(const Mat& mask, Point center, double radius, double scale = 3.0);// 손가락의 개수를 세기 위한 함수

int main_menu();// 메인메뉴
void sale_of_goods();// 상품판매
void inventory();// 재고관리

string menu[] = { "0","우유(Milk)           ","사이다(Cider)        ","막걸리(Makgeolli)    ","토레타(Toreta)       ","홍삼차(Red ginseng tea)" };// 상품현황
int stock[] = { 0,50,50,50,50,50 };// 재고현황
int finger_number=0;// 다용도로 인터페이스 받는 전역변수
VideoCapture cap;// 카메라를 다루기 위한 객체
int Menu_Number;// 메뉴의 번호
int recommendation = 0;// 상품자동추천 할지말지를 저장
String ageList[8] = { "(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)" };//연령을 8가지 범주로 구분
String genderList[2] = { "Male","Female" };// 나이의 범주

const String face_model = "opencv_face_detector_uint8.pb";// 얼굴 검출을 위한 Model파일
const String face_config = "opencv_face_detector.pbtxt";// 얼굴 검출을 위한 Config파일
const String age_model = "age_net.caffemodel";// 나이 검출을 위한 Model파일
const String age_config = "age_deploy.prototxt";// 나이 검출을 위한 Config파일
const String gender_model = "gender_net.caffemodel";// 성별 검출을 위한 Model파일
const String gender_config = "gender_deploy.prototxt";// 성별 검출을 위한 Config파일

Net face_net, age_net, gender_net;// 얼굴, 나이, 성별의 네트워크정보를 저장하는 Net객체

int main(){// 메인함수
	cap.open(0);// 0번 카메라(웹캠을 연다)
	if (!cap.isOpened()) {// 카메라가 제대로 열렸는지 확인
		cerr << "camera open failed!" << endl;//카메라가 열리지 않았다면 에러메시지 출력
		return -1;// 프로그램 종료
	}

	thread real_time(real_time_video);// 실시간 영상 출력
	thread finger_detector(finger_counting);// 실시간 영상인식을 인식하여 손가락 개수를 센다.
	thread th_age_gender(predict_age_gender);// 영상에서 나이 성별을 예측해주는 함수

	face_net = readNet(face_model, face_config);
	age_net = readNet(age_model, age_config);
	gender_net = readNet(gender_model, gender_config);
	if (face_net.empty() || age_net.empty() || gender_net.empty()) {
		cerr << "Net open failed!" << endl;
		return -1;
	}
	while (1) {// 영상인식 상품관리시스템 무한 반복
		Menu_Number = main_menu();// 메인메뉴
		if (Menu_Number == 1) {// 1번 메뉴 선택됨
			cout << "1번 상품판매 메뉴가 선택 되었습니다." << endl << endl;// 1번 메뉴 선택된 것을 출력
			sale_of_goods();// 상품판매 실행
		}
		else if (Menu_Number == 2) {// 2번 메뉴 선택됨
			cout << "2번 재고관리 메뉴가 선택되었습니다." << endl << endl;// 2번 메뉴 선택된 것을 출력
			inventory();// 재고관리 실행
		}
		else if (Menu_Number == 3) {
			cout << "3번 종료 메뉴가 선택되었습니다." << endl;//3번 메뉴가 선택된것을 출력
			cout << "프로그램을 종료합니다." << endl;// 3번 메뉴 선택되어 프로그램 종료
			waitKey(2000);// 실행파일로 실행시키면 창을 바로 닫아버려서 좀 기다려줌
			break;
		}
		else {
			cout << Menu_Number << "번은 없는 메뉴입니다. 다시 선택해주세요~" << endl << endl;// 해당메뉴가 없어서 경고출력
		}
	}
	real_time.join();// real_time 함수가 리턴되어 돌아 오는 곳
	finger_detector.join();// finger_detector 함수가 리턴되어 돌아 오는 곳
	th_age_gender.join();// th_age_gender 함수가 리턴되어 돌아 오는 곳
	return 0;// 프로그램 종료
}
String str1,str2;
int ii;
int main_menu() {// 메인 메뉴 함수
	str1 = "Recognizing...";// 성별, 나이를 예측하는 동안 보여주는 메세지
	str2 = "Please wait!!";// 성별, 나이를 예측하는 동안 보여주는 메세지
	recommendation = 0;// 상품추천기능이 선택됬는지를 판단하기위해 변수 0으로 초기화
	ii = 1;// 얼굴인식을 실시간으로 인식하지 않고 상품판매중 1번만 실행시키기 위한 변수 초기화
	cout << "======== 메인 메뉴 ========" << endl;// 메인 메뉴 화면 출력
	cout << "1. 상품판매" << endl;
	cout << "2. 재고관리" << endl;
	cout << "3. 종료" << endl;
	cout << "===========================" << endl << endl;
	cout << "어떤 업무를 하시겠습니까? 손으로 알려주세요~";
	cout << endl<<endl;
	finger_number = 0;// 손가락의 개수를 0으로 초기화
	while(finger_number==0){}// 손가락의 개수가 0이면 입력을 받지 못한 것이므로 입력을 받을때까지 대기(무한반복)
	return finger_number;// 손가락의 개수를 리턴
}

void sale_of_goods(){// 상품판매 함수
	do {
		cout << endl << "========================== 상품 판매 ==========================" << endl;//상품판매 출력
		cout << "1. 상품자동추천(사용자의 성별과 나이를 인식하여 상품을 추천해줌)" << endl;
		cout << "2. 수동 선택(상품,수량)" << endl;
		cout << "===============================================================" << endl << endl;
		cout << "상품을 추천 받기를 원하시면 1을 아니면 2를 손으로 표현하세요~" << endl<<endl;
		finger_number = 0;
		while (finger_number == 0) {}// 입력을 받을때까지 대기(무한반복)
		recommendation = finger_number;// 입력 받은 수를 recommendation에 저장
	} while (recommendation!=1&&recommendation!=2);// 3,4,5번을 입력받으면 위 상품판매 출력을 반복함
	if (recommendation == 1) {
		cout << "1번 상품자동추천기능을 선택하셨습니다."<< endl<<endl;// 상품자동추천이 선택되면 출력
		cout << "추천된 상품은 영상에서 확인하세요~" << endl;
	}
	else if(recommendation==2) cout << "2번 수동 선택기능을 선택하셨습니다."<< endl;//수동선택이 선택되면 출력
	
	int price[] = { 0,4000,4000,3500,4500,3500 };
	int amount[] = { 0,0,0,0,0,0 };
	int total_price[] = { 0,0,0,0,0,0 };
	int sum = 0; //총 금액 저장
	int pay = 0; //결제금액 저장
	int x; // 상품번호 저장
	int c; // y or n 저장 1 or 2 저장
	do {
		cout << endl << "============== 상품 메뉴 ==============" << endl;// 상품메뉴 출력
		for (int i = 1; i < 6; i++)
		{
			cout << i << ". " << menu[i] << "\t" << price[i] << "원" << endl;// 상품메뉴와 단가를 출력
		}
		cout << "=======================================" << endl << endl;
		cout << "어떤 상품을 구매하시겠습니까? 손으로 알려주세요~"<<endl<<endl;// 안내메세지 출력
		finger_number = 0;
		while(finger_number==0){}// 입력받을 때까지 무한대기
		x = finger_number;// x에 상품 번호 저장
		cout <<x<<"번 "<<menu[x] << "를 선택하셨습니다." << endl<<endl;// 선택된 상품 출력
		cout << "단가는 " << price[x] << "원입니다." << endl << endl;// 선택된 상품 단가 출력
		cout << "몇 개를 구매하시겠습니까? 손으로 알려주세요~" << endl<<endl;// 구매수량 안내메세지 출력
		finger_number = 0;
		while (finger_number == 0) {}// 입력받을때까지 무한대기
		int y = finger_number;// 입력받은 수량을 y에 저장
		cout << y << "개 주문받았습니다." << endl<<endl;// 입력받은 수량 출력
		amount[x] += y;// 입력받은 수량을 저장해둠
		total_price[x] = price[x] * amount[x];// 수량을 고려한 해당상품의 총가격저장
		cout << "구매하신 금액은 " << total_price[x] << "원입니다." << endl << endl;// 총가격 출력
		do {
			cout << "더 구매하시겠습니까? 손으로 알려주세요~(예: 1, 아니오: 2) " << endl<<endl;// 추가구매할지 물어봄
			finger_number = 0;
			while (finger_number == 0) {}// 입력받을때까지 무한대기
			c = finger_number;// 입력받은 수를 c에 저장
			if (c == 1) { cout << "1번 선택했습니다. 추가 구매를 진행합니다." << endl; }//c가1이면 추가구매 진행
			else if (c == 2) { cout << "2번 선택했습니다. 추가 구매는 없습니다." << endl; }// c가2이면 추가구매 없음
		} while (c != 1&&c != 2);// 입력을 제대로 받지 않으면 다시 반복
	} while (c == 1);// 추가구매가 선택되면 상품메뉴를 다시 보여줌
	for (int i = 1; i < 6; i++)
	{
		sum += total_price[i];// 구매한 모든 상품에 총 가격을 저장
	}
	cout << endl << "구매한신 총 금액은 " << sum << "원입니다." << endl<<endl;// 구매한 총금액을 출력
	while (1)
	{
		cout << "결제금액을 입금해주시기 바랍니다.(키보드로 입력해주세요~)" << endl;
		cout << "결제금액 : ";// 결제금액을 입력하라고 안내메세지 출력
		int n = 0;
		cin >> n;// 입력받은 돈을 n에 저장
		pay += n;// 입력받은 돈을 pay에 누적시킴
		if (pay < sum)// 총금액보다 지불한 금액이 적은경우 부족한 금액 출력
			cout << endl << "결제금액에서 " << sum - pay << "원이 부족합니다." << endl;
		else break;// 총금액보다 지불한 금액이 같거나 크면 빠져나감
	}
	cout << endl << endl;
	cout << "=============== 영 수 증 ===============" << endl;// 영수증 출력
	cout << "품목\t\t\t수량\t금액" << endl;
	for (int i = 1; i < 6; i++)
	{
		if (amount[i] != 0)// 구매한 상품과 수량, 가격 출력
		{
			cout << menu[i] << "\t" << amount[i] << "\t" << total_price[i] << "원" << endl;
		}
	}
	cout << "========================================" << endl;
	cout << "총구매금액\t\t" << sum << "원" << endl;// 총구매금액 출력
	cout << "받은금액\t\t" << pay << "원" << endl;// 받은금액 출력
	cout << "거스름돈\t\t" << pay - sum << "원" << endl;// 거스름돈 출력
	cout << "감사합니다~ 좋은 하루 되세요!" << endl;
	cout << "========================================" << endl << endl;
	for (int i = 1; i < 6; i++) {// 상품판매를 고려한 재고량 계산
		stock[i] -= amount[i];
	}
}
void inventory() {// 재고관리
	int n;
	do {
		cout << endl <<"======== 재고 관리 ========" << endl;// 재고관리 출력
		cout << "1. 재고조회" << endl;// 1번 재고조회
		cout << "2. 상품입고" << endl;// 2번 상품입고
		cout << "===========================" << endl << endl;
		cout << "어떤 업무를 하시겠습니까? 손으로 알려주세요~";// 안내 메세지 출력
		cout << endl;
		finger_number = 0;
		while (finger_number == 0) {}// 입력받을때까지 무한대기
		n = finger_number;
	} while (n != 1 && n != 2);// 제대로 입력받지않으면 위 문장 반복
	if (n == 1) {// 1번 선택됨
		cout << endl << "1번 재고조회가 선택 되었습니다." << endl;
		cout << endl << "============= 재고 조회 =============" << endl;//현재 남은 재고량을 출력
		for (int i = 1; i < 6; i++)
		{
			cout << i << ". " << menu[i] << "\t" << stock[i] << "개" << endl;// 상품과 재고량 출력
		}
		cout << "=====================================" << endl << endl;
	}
	else if (n == 2) {// 2번 선택됨
		cout << endl << "2번 상품입고가 선택 되었습니다." << endl;
		int x = 0;
		int y = 0;
		cout << endl << "======== 상품 입고 ========" << endl;// 상품입고
		for (int i = 1; i < 6; i++)// 상품번호와 상품메뉴 출력
		{
			cout << i << ". " << menu[i] << endl;
		}
		cout << "===========================" << endl << endl;
		cout << "입고할 상품을 손으로 알려주세요~"<<endl<<endl;
		finger_number = 0;
		while (finger_number == 0) {}//입력받을 때까지 무한대기
		x = finger_number;
		cout <<x <<"번 "<<menu[x] << "를 선택하셨습니다." << endl<<endl;//선택한 상품번호와 상품 출력
		cout << "입고 수량을 손으로 알려주세요~"<<endl<<endl;
		finger_number = 0;
		while (finger_number == 0) {}// 입력받을 때까지 무한대기
		y = finger_number;//입고수량 저장
		cout << y << "개를 입고합니다." << endl<<endl;// 입력받은 수량 출력
		stock[x] += y;// 입고된 수량을 재고량에 누적시킴
		cout << "입고가 완료되었습니다." << endl << endl;
	}
}

double radius;// 손바닥영역의 반지름
Mat mask;// 손이 검출된 이진화 영상을 저장할 Mat 객체
Point center;// 손바닥의 중심점
Mat frame;// frame 을 저장할 객체
int z = 0;// 예비 변수


void real_time_video() {// 실시간 카메라 영상을 보여주는 함수
	double delay = 1000 / cap.get(CAP_PROP_FPS);// 정지영상을 보여주는 시간 설정
	Mat roi;// 관심영역을 저장할 객체
	while (1) {
		if (Menu_Number == 3) return;
		cap >> frame;// frame을 1장 가져옴
		if (frame.empty()) break;// frame이 비어있으면 break
		if (recommendation == 1) {// 상품자동추천기능이 실행되면 화면 하단에 인식한 성별,나이,추천한 상품 보여줌
			putText(frame, str1, Point(2, 340), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
			putText(frame, str2, Point(2, 390), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
		}
		rectangle(frame, Rect(Point(10, 10), Point(320, 240)), Scalar(0, 255, 0), 5);// 손을 검출할 영역을 보여줌
		roi = frame(Rect(Point(10, 10), Point(320, 240)));// 손을 검출할 영역을 관심영역으 지정
		mask = getHandMask1(&roi, 133, 173, 77, 127);// 손을 검출
		center = getHandCenter(mask, radius);// 검출된 손으로 부터 손바닥의 중심점과 손바닥영역의 반지름을 구함
		if (radius != 0) {// 손이 검출이 되었을때 손바닥의 중심, 손바닥영역, 손가락을 세기위한 원을 그려줌
			circle(frame, center, 2, Scalar(0, 255, 0), -1);// 손바닥의 중심을 그려줌
			circle(frame, center, (int)(radius + 0.5), Scalar(255, 0, 0), 2);// 손바닥 영역의 원을 그려줌
			circle(frame, center, 106, Scalar(0, 0, 255), 2);// 손가락을 세기위한 원을 그려줌
			putText(frame, format("Finger Counting: %d", z), Point(10, 300), FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 255), 2);
		}
		imshow("frame", frame);// 카메라로 찍힌 영상을 출력
		imshow("dst", mask);// 관심영역 영상을 출력
		if (waitKey(delay) == 27) break;// esc키를 누르면 break
	}
}

void finger_counting()// 영상인식 손가락 개수 세기
{
	while (1) {
		if (Menu_Number == 3) return;// Menu_Number가 3이면 프로그램 종료를 의미한다.
		finger_number = 0;// 손가락을 세기전에 0으로 초기화 해준다.
		if (radius != 0) {// radius가 0일때는 손이 검출되지 않았다는 것이다.
			waitKey(1300);// 일단 손이 검출되면 1초를 대기한다.(손이 갑자기 들어오면 잘못 인식 할수도 있기때문)
			finger_number = getFingerCount(mask, center, radius);// 검출된 손의 이진영상으로 부터 손가락 개수를 센다.
			z = finger_number;// 손가락 개수를 전역변수 z에 저장한다.
			waitKey(1500);// 손가락의 개수를 세고 1.5초대기한다.(대기하지않으면 다음 선택도 결정될 수 있음)
		}
	}
}

Mat image, YCrCb;// 여기 변수들은 getHandMask1함수에서 사용되는되 
vector<Mat> planes;// 손검출은 계속 수시로 수행되는 거여서 
int num, num1;// getHandMask1함수에서 계속 변수선언하는게 싫어서 전역변수로 선언한거다.

Mat getHandMask1(void* src, int minCr, int maxCr, int minCb, int maxCb) {// 손검출함수
	image = *(Mat*)src;// 손을 검출할 영역의 Mat객체를 image에 저장
	cvtColor(image, YCrCb, COLOR_BGR2YCrCb);//컬러 공간 변환 BGR->YCrCb
	split(YCrCb, planes);//각 채널별로 분리
	Mat mask(image.size(), CV_8UC1, Scalar(0));//손을 검출한 이진영상을 저장할 객체
	for (int i = 0; i < image.rows; i++) {//cr채널과 cb채널 화서처리
		for (int j = 0; j < image.cols; j++) {
			num = planes[1].at<uchar>(i, j);// cr채널의 화소값
			num1 = planes[2].at<uchar>(i, j);// cb채널의 화소값
			if ((minCr <= num && num <= maxCr) && (minCb <= num1 && num1 <= maxCb))
				mask.at<uchar>(i, j) = 255;// cr,cb채널의 화소값이 피부색 범위안에 있으면 
		}                                  // mask에 좌표에 255을 저장
	}
	erode(mask, mask, Mat(3, 3, CV_8UC1, Scalar(1)), Point(-1, -1), 6);//검출된 손영상에서 잡음제거를 위한 침식연산
	//blur(mask, mask, Size(3, 3));// blur는 적용하지 않아도 잘 인식되어 사용하지 않음
	return mask;// 검출된 손 이진영상을 리턴
}

Point getHandCenter(const Mat& mask, double& radius) {//손바닥의 중심점과 반지름을 구하는 함수
	Mat dst;// 거리변환행렬을 저장할 객체
	distanceTransform(mask, dst, DIST_L2, 5);//mask를 거리변환행렬로 변환해서 dst에저장
	Point pt;//손바닥의 중심좌표를 저장할 객체
	minMaxLoc(dst, 0, &radius, 0, &pt);// 거리변환행렬에서 가장 큰 값을 가지는 좌표가 손바닥의 중심점임
	return pt;// 손바닥의 중심좌표를 리턴
}// 검출된 손영상과 중심점,반지름을 받아 손가락의 개수를 센다.
int getFingerCount(const Mat& mask, Point center, double radius, double scale) {
	Mat clmg(mask.size(), CV_8UC1, Scalar(0)); //0으로초기환된 mask사이즈의 객체생성  
	int fingerCount = 0;// 손가락개수를 0으로 초기화한다.
	if (radius == 0) return 0;// 손을 검출하지 못했다면 0을 리턴
	circle(clmg, center, 106, Scalar(255));// 손가락을 세기위한 center가 중심점이고 radius보다 큰 반지름의 원을 만듬
	vector<vector<Point>> contours;// contours는 위에서 만든 원의 모든 좌표를 저장한다.
	findContours(clmg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);// cImg에서 컨투어를 찾아 컨투어의 모든 좌표 저장
	if (contours.size() == 0) return 0;//컨투어를 찾지 못했다면 0을 리턴
	for (int i = 1; i < contours[0].size(); i++) {//원을 따라가면서 값이 0에서 255로 바뀌는 부분의 수를 센다.
		Point p1 = contours[0][i - 1];// 바로 전 좌표
		Point p2 = contours[0][i];// 현재 좌표
		if (mask.at<uchar>(p1.y, p1.x) == 0 && mask.at<uchar>(p2.y, p2.x) > 1) {
			fingerCount++;// 원을 따라가면서 픽셀값이 0이었다가 그 다음 픽셀값이 255가 되면 1증가시킨다.
		}
	}
	return fingerCount / 2;// 이론적으로는 fingerCount를 리턴해야 맞지만 손 검출된 영상의 특성상 2로 나눠서 리턴해줌
}// 사람눈으로는 픽셀값이 255에서 0으로 바뀌는듯 한데 자세히 보면 0에서 255로 바뀌는 부분이 있는 듯하다.

void predict_age_gender() {// 나이, 성별 예측 함수
	while (1) {
		if (Menu_Number == 3) return;// Menu_Number가 3이면 프로그램 종료
		if (recommendation == 1) {// 상품자동 추천기능이 실행되면 실행
			if (ii) {// 상품판매가 진행되고 있는 중에는 한번만 실행되기 위해서 ii를 통해 구현함
				waitKey(2000);// 상품추천이 진행되는 동안 기다리라는 메세지를 보여주려고 좀 기다려 줬음
				Mat roi_face;// 얼굴영역을 저장할 객체
				Mat face_blob = blobFromImage(frame, 1, Size(300, 300), Scalar(104, 177, 123));// 블롭으로 바꿈
				face_net.setInput(face_blob);// 얼굴검출을 위한 네트워크에 입력
				Mat face_res = face_net.forward();// 예측을 수행함
				Mat face_detect(face_res.size[2], face_res.size[3], CV_32FC1, face_res.ptr<float>());//예측된결과저장
					for (int i = 0; i < face_detect.rows; i++) {//검출된 얼굴의 개수만큼 반복
						float confidence = face_detect.at<float>(i, 2);//얼굴일 확률을 confidence에 저장
						if (confidence < 0.5) break;//확률이 50%미만이면 break
						int face_x1 = cvRound(face_detect.at<float>(i, 3)*frame.cols);//얼굴영역의 좌측상단 
						int face_y1 = cvRound(face_detect.at<float>(i, 4)*frame.rows);// x,y좌표
						int face_x2 = cvRound(face_detect.at<float>(i, 5)*frame.cols);//얼굴영역의 우측하단
						int face_y2 = cvRound(face_detect.at<float>(i, 6)*frame.rows);//x,y좌표
						roi_face = frame(Rect(Point(face_x1 - 25, face_y1 - 25), Point(face_x2 + 25, face_y2 + 25)));
					}// 영상에 사람은 1명이라고 가정하고 프로그램을 짰다.
					Scalar mean(78.4263377603, 87.7689143744, 114.895847746);
					Mat age_gender_blob = blobFromImage(roi_face, 1.0f, Size(227, 227), mean);//얼굴영역을 blob 변환
					gender_net.setInput(age_gender_blob);// gender_net에 입력함
					Mat gender_res = gender_net.forward();// 성별예측을 수행함
					double max1, max2;// 예측된 결과의 최대값을 저장할 변수
					Point ptmax1, ptmax2;// 예측된 결과의 최대값의 좌표를 저장할 객체
					minMaxLoc(gender_res, 0, &max1, 0, &ptmax1);// 성별을 예측 결과를 ptmax1를 통해 알수있음
					age_net.setInput(age_gender_blob);// age_net에 입력함
					Mat age_res = age_net.forward();// 나이 예측을 수행함
					minMaxLoc(age_res, 0, &max2, 0, &ptmax2);// 나이예측의 결과를 ptmax2를 통해 알수있음
					str1 = format("gender: %s, age: %s", genderList[ptmax1.x].c_str(), ageList[ptmax2.x].c_str());
					int predict_age, predict_gender;// str1에는 예측한 성별과 나이를 문자열로 저장
					predict_gender = ptmax1.x;// 성별을 예측한 결과정보를 저장
					predict_age = ptmax2.x;// 나이를 예측한 결과정보를 저장
					if (predict_age == 0 || predict_age == 1)	str2 = "I recommend Milk";// 성별과나이에 따라
					else if (predict_age == 2 || (predict_gender == 0 && predict_age == 3)) // str2에 추천할
						str2 = "I recommend Cider";                                         //상품이 문자열로 
					else if (predict_gender == 0 && (predict_age == 4 || predict_age == 5)) // 저장됨
						str2 = "I recommend Makgeolli";
					else if (predict_gender == 1 && (predict_age == 3 || predict_age == 4 || predict_age == 5 || predict_age == 6))	 str2 = "I recommend Toreta";
					else if (predict_age == 7 || (predict_gender == 0 && predict_age == 6)) 
						str2 = "I recommend Red ginseng tea";
					ii = 0;
			}
		}
	}
}