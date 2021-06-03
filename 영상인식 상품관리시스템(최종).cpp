#include<iostream>// c++���� �������
#include<opencv2/opencv.hpp>// opencv �������
#include<thread>// ������ ���� �������

using namespace std;// ���ӽ����̽� std ����
using namespace cv;// ���ӽ����̽� cv ����
using namespace cv::dnn;// ���ӽ����̽� dnn����

void real_time_video();// �ǽð� ī�޶� ������ ����ϴ� �Լ�
void finger_counting();// �ǽð� �հ��� ���� ���� �Լ�
void predict_age_gender();// ����, ���� ���� �Լ�

// ����(0-2): ����, ����(4-6): ����, ����(8-12): ���̴�, 
// ����(15-20): ���̴�, ����(25-32): ���ɸ�, ����(48-53): ȫ����, ����(60-100): ȫ����
// ����(0-2): ����, ����(4-6): ����, ����(8-12): ���̴�, 
// ����(15-20): �䷹Ÿ, ����(25-32): �䷹Ÿ, ����(48-53): �䷹Ÿ, ����(60-100): ȫ����

Mat getHandMask1(void* src, int minCr, int maxCr, int minCb, int maxCb);// ���� �����ϱ����� �Լ�
Point getHandCenter(const Mat& mask, double& radius);// �չٴ��� �߽���,�չٴ� ������ �������� ���ϱ� ���� �Լ�
int getFingerCount(const Mat& mask, Point center, double radius, double scale = 3.0);// �հ����� ������ ���� ���� �Լ�

int main_menu();// ���θ޴�
void sale_of_goods();// ��ǰ�Ǹ�
void inventory();// ������

string menu[] = { "0","����(Milk)           ","���̴�(Cider)        ","���ɸ�(Makgeolli)    ","�䷹Ÿ(Toreta)       ","ȫ����(Red ginseng tea)" };// ��ǰ��Ȳ
int stock[] = { 0,50,50,50,50,50 };// �����Ȳ
int finger_number=0;// �ٿ뵵�� �������̽� �޴� ��������
VideoCapture cap;// ī�޶� �ٷ�� ���� ��ü
int Menu_Number;// �޴��� ��ȣ
int recommendation = 0;// ��ǰ�ڵ���õ ���������� ����
String ageList[8] = { "(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)" };//������ 8���� ���ַ� ����
String genderList[2] = { "Male","Female" };// ������ ����

const String face_model = "opencv_face_detector_uint8.pb";// �� ������ ���� Model����
const String face_config = "opencv_face_detector.pbtxt";// �� ������ ���� Config����
const String age_model = "age_net.caffemodel";// ���� ������ ���� Model����
const String age_config = "age_deploy.prototxt";// ���� ������ ���� Config����
const String gender_model = "gender_net.caffemodel";// ���� ������ ���� Model����
const String gender_config = "gender_deploy.prototxt";// ���� ������ ���� Config����

Net face_net, age_net, gender_net;// ��, ����, ������ ��Ʈ��ũ������ �����ϴ� Net��ü

int main(){// �����Լ�
	cap.open(0);// 0�� ī�޶�(��ķ�� ����)
	if (!cap.isOpened()) {// ī�޶� ����� ���ȴ��� Ȯ��
		cerr << "camera open failed!" << endl;//ī�޶� ������ �ʾҴٸ� �����޽��� ���
		return -1;// ���α׷� ����
	}

	thread real_time(real_time_video);// �ǽð� ���� ���
	thread finger_detector(finger_counting);// �ǽð� �����ν��� �ν��Ͽ� �հ��� ������ ����.
	thread th_age_gender(predict_age_gender);// ���󿡼� ���� ������ �������ִ� �Լ�

	face_net = readNet(face_model, face_config);
	age_net = readNet(age_model, age_config);
	gender_net = readNet(gender_model, gender_config);
	if (face_net.empty() || age_net.empty() || gender_net.empty()) {
		cerr << "Net open failed!" << endl;
		return -1;
	}
	while (1) {// �����ν� ��ǰ�����ý��� ���� �ݺ�
		Menu_Number = main_menu();// ���θ޴�
		if (Menu_Number == 1) {// 1�� �޴� ���õ�
			cout << "1�� ��ǰ�Ǹ� �޴��� ���� �Ǿ����ϴ�." << endl << endl;// 1�� �޴� ���õ� ���� ���
			sale_of_goods();// ��ǰ�Ǹ� ����
		}
		else if (Menu_Number == 2) {// 2�� �޴� ���õ�
			cout << "2�� ������ �޴��� ���õǾ����ϴ�." << endl << endl;// 2�� �޴� ���õ� ���� ���
			inventory();// ������ ����
		}
		else if (Menu_Number == 3) {
			cout << "3�� ���� �޴��� ���õǾ����ϴ�." << endl;//3�� �޴��� ���õȰ��� ���
			cout << "���α׷��� �����մϴ�." << endl;// 3�� �޴� ���õǾ� ���α׷� ����
			waitKey(2000);// �������Ϸ� �����Ű�� â�� �ٷ� �ݾƹ����� �� ��ٷ���
			break;
		}
		else {
			cout << Menu_Number << "���� ���� �޴��Դϴ�. �ٽ� �������ּ���~" << endl << endl;// �ش�޴��� ��� ������
		}
	}
	real_time.join();// real_time �Լ��� ���ϵǾ� ���� ���� ��
	finger_detector.join();// finger_detector �Լ��� ���ϵǾ� ���� ���� ��
	th_age_gender.join();// th_age_gender �Լ��� ���ϵǾ� ���� ���� ��
	return 0;// ���α׷� ����
}
String str1,str2;
int ii;
int main_menu() {// ���� �޴� �Լ�
	str1 = "Recognizing...";// ����, ���̸� �����ϴ� ���� �����ִ� �޼���
	str2 = "Please wait!!";// ����, ���̸� �����ϴ� ���� �����ִ� �޼���
	recommendation = 0;// ��ǰ��õ����� ���É������ �Ǵ��ϱ����� ���� 0���� �ʱ�ȭ
	ii = 1;// ���ν��� �ǽð����� �ν����� �ʰ� ��ǰ�Ǹ��� 1���� �����Ű�� ���� ���� �ʱ�ȭ
	cout << "======== ���� �޴� ========" << endl;// ���� �޴� ȭ�� ���
	cout << "1. ��ǰ�Ǹ�" << endl;
	cout << "2. ������" << endl;
	cout << "3. ����" << endl;
	cout << "===========================" << endl << endl;
	cout << "� ������ �Ͻðڽ��ϱ�? ������ �˷��ּ���~";
	cout << endl<<endl;
	finger_number = 0;// �հ����� ������ 0���� �ʱ�ȭ
	while(finger_number==0){}// �հ����� ������ 0�̸� �Է��� ���� ���� ���̹Ƿ� �Է��� ���������� ���(���ѹݺ�)
	return finger_number;// �հ����� ������ ����
}

void sale_of_goods(){// ��ǰ�Ǹ� �Լ�
	do {
		cout << endl << "========================== ��ǰ �Ǹ� ==========================" << endl;//��ǰ�Ǹ� ���
		cout << "1. ��ǰ�ڵ���õ(������� ������ ���̸� �ν��Ͽ� ��ǰ�� ��õ����)" << endl;
		cout << "2. ���� ����(��ǰ,����)" << endl;
		cout << "===============================================================" << endl << endl;
		cout << "��ǰ�� ��õ �ޱ⸦ ���Ͻø� 1�� �ƴϸ� 2�� ������ ǥ���ϼ���~" << endl<<endl;
		finger_number = 0;
		while (finger_number == 0) {}// �Է��� ���������� ���(���ѹݺ�)
		recommendation = finger_number;// �Է� ���� ���� recommendation�� ����
	} while (recommendation!=1&&recommendation!=2);// 3,4,5���� �Է¹����� �� ��ǰ�Ǹ� ����� �ݺ���
	if (recommendation == 1) {
		cout << "1�� ��ǰ�ڵ���õ����� �����ϼ̽��ϴ�."<< endl<<endl;// ��ǰ�ڵ���õ�� ���õǸ� ���
		cout << "��õ�� ��ǰ�� ���󿡼� Ȯ���ϼ���~" << endl;
	}
	else if(recommendation==2) cout << "2�� ���� ���ñ���� �����ϼ̽��ϴ�."<< endl;//���������� ���õǸ� ���
	
	int price[] = { 0,4000,4000,3500,4500,3500 };
	int amount[] = { 0,0,0,0,0,0 };
	int total_price[] = { 0,0,0,0,0,0 };
	int sum = 0; //�� �ݾ� ����
	int pay = 0; //�����ݾ� ����
	int x; // ��ǰ��ȣ ����
	int c; // y or n ���� 1 or 2 ����
	do {
		cout << endl << "============== ��ǰ �޴� ==============" << endl;// ��ǰ�޴� ���
		for (int i = 1; i < 6; i++)
		{
			cout << i << ". " << menu[i] << "\t" << price[i] << "��" << endl;// ��ǰ�޴��� �ܰ��� ���
		}
		cout << "=======================================" << endl << endl;
		cout << "� ��ǰ�� �����Ͻðڽ��ϱ�? ������ �˷��ּ���~"<<endl<<endl;// �ȳ��޼��� ���
		finger_number = 0;
		while(finger_number==0){}// �Է¹��� ������ ���Ѵ��
		x = finger_number;// x�� ��ǰ ��ȣ ����
		cout <<x<<"�� "<<menu[x] << "�� �����ϼ̽��ϴ�." << endl<<endl;// ���õ� ��ǰ ���
		cout << "�ܰ��� " << price[x] << "���Դϴ�." << endl << endl;// ���õ� ��ǰ �ܰ� ���
		cout << "�� ���� �����Ͻðڽ��ϱ�? ������ �˷��ּ���~" << endl<<endl;// ���ż��� �ȳ��޼��� ���
		finger_number = 0;
		while (finger_number == 0) {}// �Է¹��������� ���Ѵ��
		int y = finger_number;// �Է¹��� ������ y�� ����
		cout << y << "�� �ֹ��޾ҽ��ϴ�." << endl<<endl;// �Է¹��� ���� ���
		amount[x] += y;// �Է¹��� ������ �����ص�
		total_price[x] = price[x] * amount[x];// ������ ����� �ش��ǰ�� �Ѱ�������
		cout << "�����Ͻ� �ݾ��� " << total_price[x] << "���Դϴ�." << endl << endl;// �Ѱ��� ���
		do {
			cout << "�� �����Ͻðڽ��ϱ�? ������ �˷��ּ���~(��: 1, �ƴϿ�: 2) " << endl<<endl;// �߰��������� ���
			finger_number = 0;
			while (finger_number == 0) {}// �Է¹��������� ���Ѵ��
			c = finger_number;// �Է¹��� ���� c�� ����
			if (c == 1) { cout << "1�� �����߽��ϴ�. �߰� ���Ÿ� �����մϴ�." << endl; }//c��1�̸� �߰����� ����
			else if (c == 2) { cout << "2�� �����߽��ϴ�. �߰� ���Ŵ� �����ϴ�." << endl; }// c��2�̸� �߰����� ����
		} while (c != 1&&c != 2);// �Է��� ����� ���� ������ �ٽ� �ݺ�
	} while (c == 1);// �߰����Ű� ���õǸ� ��ǰ�޴��� �ٽ� ������
	for (int i = 1; i < 6; i++)
	{
		sum += total_price[i];// ������ ��� ��ǰ�� �� ������ ����
	}
	cout << endl << "�����ѽ� �� �ݾ��� " << sum << "���Դϴ�." << endl<<endl;// ������ �ѱݾ��� ���
	while (1)
	{
		cout << "�����ݾ��� �Ա����ֽñ� �ٶ��ϴ�.(Ű����� �Է����ּ���~)" << endl;
		cout << "�����ݾ� : ";// �����ݾ��� �Է��϶�� �ȳ��޼��� ���
		int n = 0;
		cin >> n;// �Է¹��� ���� n�� ����
		pay += n;// �Է¹��� ���� pay�� ������Ŵ
		if (pay < sum)// �ѱݾ׺��� ������ �ݾ��� ������� ������ �ݾ� ���
			cout << endl << "�����ݾ׿��� " << sum - pay << "���� �����մϴ�." << endl;
		else break;// �ѱݾ׺��� ������ �ݾ��� ���ų� ũ�� ��������
	}
	cout << endl << endl;
	cout << "=============== �� �� �� ===============" << endl;// ������ ���
	cout << "ǰ��\t\t\t����\t�ݾ�" << endl;
	for (int i = 1; i < 6; i++)
	{
		if (amount[i] != 0)// ������ ��ǰ�� ����, ���� ���
		{
			cout << menu[i] << "\t" << amount[i] << "\t" << total_price[i] << "��" << endl;
		}
	}
	cout << "========================================" << endl;
	cout << "�ѱ��űݾ�\t\t" << sum << "��" << endl;// �ѱ��űݾ� ���
	cout << "�����ݾ�\t\t" << pay << "��" << endl;// �����ݾ� ���
	cout << "�Ž�����\t\t" << pay - sum << "��" << endl;// �Ž����� ���
	cout << "�����մϴ�~ ���� �Ϸ� �Ǽ���!" << endl;
	cout << "========================================" << endl << endl;
	for (int i = 1; i < 6; i++) {// ��ǰ�ǸŸ� ����� ��� ���
		stock[i] -= amount[i];
	}
}
void inventory() {// ������
	int n;
	do {
		cout << endl <<"======== ��� ���� ========" << endl;// ������ ���
		cout << "1. �����ȸ" << endl;// 1�� �����ȸ
		cout << "2. ��ǰ�԰�" << endl;// 2�� ��ǰ�԰�
		cout << "===========================" << endl << endl;
		cout << "� ������ �Ͻðڽ��ϱ�? ������ �˷��ּ���~";// �ȳ� �޼��� ���
		cout << endl;
		finger_number = 0;
		while (finger_number == 0) {}// �Է¹��������� ���Ѵ��
		n = finger_number;
	} while (n != 1 && n != 2);// ����� �Է¹��������� �� ���� �ݺ�
	if (n == 1) {// 1�� ���õ�
		cout << endl << "1�� �����ȸ�� ���� �Ǿ����ϴ�." << endl;
		cout << endl << "============= ��� ��ȸ =============" << endl;//���� ���� ����� ���
		for (int i = 1; i < 6; i++)
		{
			cout << i << ". " << menu[i] << "\t" << stock[i] << "��" << endl;// ��ǰ�� ��� ���
		}
		cout << "=====================================" << endl << endl;
	}
	else if (n == 2) {// 2�� ���õ�
		cout << endl << "2�� ��ǰ�԰� ���� �Ǿ����ϴ�." << endl;
		int x = 0;
		int y = 0;
		cout << endl << "======== ��ǰ �԰� ========" << endl;// ��ǰ�԰�
		for (int i = 1; i < 6; i++)// ��ǰ��ȣ�� ��ǰ�޴� ���
		{
			cout << i << ". " << menu[i] << endl;
		}
		cout << "===========================" << endl << endl;
		cout << "�԰��� ��ǰ�� ������ �˷��ּ���~"<<endl<<endl;
		finger_number = 0;
		while (finger_number == 0) {}//�Է¹��� ������ ���Ѵ��
		x = finger_number;
		cout <<x <<"�� "<<menu[x] << "�� �����ϼ̽��ϴ�." << endl<<endl;//������ ��ǰ��ȣ�� ��ǰ ���
		cout << "�԰� ������ ������ �˷��ּ���~"<<endl<<endl;
		finger_number = 0;
		while (finger_number == 0) {}// �Է¹��� ������ ���Ѵ��
		y = finger_number;//�԰���� ����
		cout << y << "���� �԰��մϴ�." << endl<<endl;// �Է¹��� ���� ���
		stock[x] += y;// �԰�� ������ ����� ������Ŵ
		cout << "�԰� �Ϸ�Ǿ����ϴ�." << endl << endl;
	}
}

double radius;// �չٴڿ����� ������
Mat mask;// ���� ����� ����ȭ ������ ������ Mat ��ü
Point center;// �չٴ��� �߽���
Mat frame;// frame �� ������ ��ü
int z = 0;// ���� ����


void real_time_video() {// �ǽð� ī�޶� ������ �����ִ� �Լ�
	double delay = 1000 / cap.get(CAP_PROP_FPS);// ���������� �����ִ� �ð� ����
	Mat roi;// ���ɿ����� ������ ��ü
	while (1) {
		if (Menu_Number == 3) return;
		cap >> frame;// frame�� 1�� ������
		if (frame.empty()) break;// frame�� ��������� break
		if (recommendation == 1) {// ��ǰ�ڵ���õ����� ����Ǹ� ȭ�� �ϴܿ� �ν��� ����,����,��õ�� ��ǰ ������
			putText(frame, str1, Point(2, 340), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
			putText(frame, str2, Point(2, 390), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
		}
		rectangle(frame, Rect(Point(10, 10), Point(320, 240)), Scalar(0, 255, 0), 5);// ���� ������ ������ ������
		roi = frame(Rect(Point(10, 10), Point(320, 240)));// ���� ������ ������ ���ɿ����� ����
		mask = getHandMask1(&roi, 133, 173, 77, 127);// ���� ����
		center = getHandCenter(mask, radius);// ����� ������ ���� �չٴ��� �߽����� �չٴڿ����� �������� ����
		if (radius != 0) {// ���� ������ �Ǿ����� �չٴ��� �߽�, �չٴڿ���, �հ����� �������� ���� �׷���
			circle(frame, center, 2, Scalar(0, 255, 0), -1);// �չٴ��� �߽��� �׷���
			circle(frame, center, (int)(radius + 0.5), Scalar(255, 0, 0), 2);// �չٴ� ������ ���� �׷���
			circle(frame, center, 106, Scalar(0, 0, 255), 2);// �հ����� �������� ���� �׷���
			putText(frame, format("Finger Counting: %d", z), Point(10, 300), FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 255), 2);
		}
		imshow("frame", frame);// ī�޶�� ���� ������ ���
		imshow("dst", mask);// ���ɿ��� ������ ���
		if (waitKey(delay) == 27) break;// escŰ�� ������ break
	}
}

void finger_counting()// �����ν� �հ��� ���� ����
{
	while (1) {
		if (Menu_Number == 3) return;// Menu_Number�� 3�̸� ���α׷� ���Ḧ �ǹ��Ѵ�.
		finger_number = 0;// �հ����� �������� 0���� �ʱ�ȭ ���ش�.
		if (radius != 0) {// radius�� 0�϶��� ���� ������� �ʾҴٴ� ���̴�.
			waitKey(1300);// �ϴ� ���� ����Ǹ� 1�ʸ� ����Ѵ�.(���� ���ڱ� ������ �߸� �ν� �Ҽ��� �ֱ⶧��)
			finger_number = getFingerCount(mask, center, radius);// ����� ���� ������������ ���� �հ��� ������ ����.
			z = finger_number;// �հ��� ������ �������� z�� �����Ѵ�.
			waitKey(1500);// �հ����� ������ ���� 1.5�ʴ���Ѵ�.(������������� ���� ���õ� ������ �� ����)
		}
	}
}

Mat image, YCrCb;// ���� �������� getHandMask1�Լ����� ���Ǵµ� 
vector<Mat> planes;// �հ����� ��� ���÷� ����Ǵ� �ſ��� 
int num, num1;// getHandMask1�Լ����� ��� ���������ϴ°� �Ⱦ ���������� �����ѰŴ�.

Mat getHandMask1(void* src, int minCr, int maxCr, int minCb, int maxCb) {// �հ����Լ�
	image = *(Mat*)src;// ���� ������ ������ Mat��ü�� image�� ����
	cvtColor(image, YCrCb, COLOR_BGR2YCrCb);//�÷� ���� ��ȯ BGR->YCrCb
	split(YCrCb, planes);//�� ä�κ��� �и�
	Mat mask(image.size(), CV_8UC1, Scalar(0));//���� ������ ���������� ������ ��ü
	for (int i = 0; i < image.rows; i++) {//crä�ΰ� cbä�� ȭ��ó��
		for (int j = 0; j < image.cols; j++) {
			num = planes[1].at<uchar>(i, j);// crä���� ȭ�Ұ�
			num1 = planes[2].at<uchar>(i, j);// cbä���� ȭ�Ұ�
			if ((minCr <= num && num <= maxCr) && (minCb <= num1 && num1 <= maxCb))
				mask.at<uchar>(i, j) = 255;// cr,cbä���� ȭ�Ұ��� �Ǻλ� �����ȿ� ������ 
		}                                  // mask�� ��ǥ�� 255�� ����
	}
	erode(mask, mask, Mat(3, 3, CV_8UC1, Scalar(1)), Point(-1, -1), 6);//����� �տ��󿡼� �������Ÿ� ���� ħ�Ŀ���
	//blur(mask, mask, Size(3, 3));// blur�� �������� �ʾƵ� �� �νĵǾ� ������� ����
	return mask;// ����� �� ���������� ����
}

Point getHandCenter(const Mat& mask, double& radius) {//�չٴ��� �߽����� �������� ���ϴ� �Լ�
	Mat dst;// �Ÿ���ȯ����� ������ ��ü
	distanceTransform(mask, dst, DIST_L2, 5);//mask�� �Ÿ���ȯ��ķ� ��ȯ�ؼ� dst������
	Point pt;//�չٴ��� �߽���ǥ�� ������ ��ü
	minMaxLoc(dst, 0, &radius, 0, &pt);// �Ÿ���ȯ��Ŀ��� ���� ū ���� ������ ��ǥ�� �չٴ��� �߽�����
	return pt;// �չٴ��� �߽���ǥ�� ����
}// ����� �տ���� �߽���,�������� �޾� �հ����� ������ ����.
int getFingerCount(const Mat& mask, Point center, double radius, double scale) {
	Mat clmg(mask.size(), CV_8UC1, Scalar(0)); //0�����ʱ�ȯ�� mask�������� ��ü����  
	int fingerCount = 0;// �հ��������� 0���� �ʱ�ȭ�Ѵ�.
	if (radius == 0) return 0;// ���� �������� ���ߴٸ� 0�� ����
	circle(clmg, center, 106, Scalar(255));// �հ����� �������� center�� �߽����̰� radius���� ū �������� ���� ����
	vector<vector<Point>> contours;// contours�� ������ ���� ���� ��� ��ǥ�� �����Ѵ�.
	findContours(clmg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);// cImg���� ����� ã�� �������� ��� ��ǥ ����
	if (contours.size() == 0) return 0;//����� ã�� ���ߴٸ� 0�� ����
	for (int i = 1; i < contours[0].size(); i++) {//���� ���󰡸鼭 ���� 0���� 255�� �ٲ�� �κ��� ���� ����.
		Point p1 = contours[0][i - 1];// �ٷ� �� ��ǥ
		Point p2 = contours[0][i];// ���� ��ǥ
		if (mask.at<uchar>(p1.y, p1.x) == 0 && mask.at<uchar>(p2.y, p2.x) > 1) {
			fingerCount++;// ���� ���󰡸鼭 �ȼ����� 0�̾��ٰ� �� ���� �ȼ����� 255�� �Ǹ� 1������Ų��.
		}
	}
	return fingerCount / 2;// �̷������δ� fingerCount�� �����ؾ� ������ �� ����� ������ Ư���� 2�� ������ ��������
}// ��������δ� �ȼ����� 255���� 0���� �ٲ�µ� �ѵ� �ڼ��� ���� 0���� 255�� �ٲ�� �κ��� �ִ� ���ϴ�.

void predict_age_gender() {// ����, ���� ���� �Լ�
	while (1) {
		if (Menu_Number == 3) return;// Menu_Number�� 3�̸� ���α׷� ����
		if (recommendation == 1) {// ��ǰ�ڵ� ��õ����� ����Ǹ� ����
			if (ii) {// ��ǰ�ǸŰ� ����ǰ� �ִ� �߿��� �ѹ��� ����Ǳ� ���ؼ� ii�� ���� ������
				waitKey(2000);// ��ǰ��õ�� ����Ǵ� ���� ��ٸ���� �޼����� �����ַ��� �� ��ٷ� ����
				Mat roi_face;// �󱼿����� ������ ��ü
				Mat face_blob = blobFromImage(frame, 1, Size(300, 300), Scalar(104, 177, 123));// ������� �ٲ�
				face_net.setInput(face_blob);// �󱼰����� ���� ��Ʈ��ũ�� �Է�
				Mat face_res = face_net.forward();// ������ ������
				Mat face_detect(face_res.size[2], face_res.size[3], CV_32FC1, face_res.ptr<float>());//�����Ȱ������
					for (int i = 0; i < face_detect.rows; i++) {//����� ���� ������ŭ �ݺ�
						float confidence = face_detect.at<float>(i, 2);//���� Ȯ���� confidence�� ����
						if (confidence < 0.5) break;//Ȯ���� 50%�̸��̸� break
						int face_x1 = cvRound(face_detect.at<float>(i, 3)*frame.cols);//�󱼿����� ������� 
						int face_y1 = cvRound(face_detect.at<float>(i, 4)*frame.rows);// x,y��ǥ
						int face_x2 = cvRound(face_detect.at<float>(i, 5)*frame.cols);//�󱼿����� �����ϴ�
						int face_y2 = cvRound(face_detect.at<float>(i, 6)*frame.rows);//x,y��ǥ
						roi_face = frame(Rect(Point(face_x1 - 25, face_y1 - 25), Point(face_x2 + 25, face_y2 + 25)));
					}// ���� ����� 1���̶�� �����ϰ� ���α׷��� ®��.
					Scalar mean(78.4263377603, 87.7689143744, 114.895847746);
					Mat age_gender_blob = blobFromImage(roi_face, 1.0f, Size(227, 227), mean);//�󱼿����� blob ��ȯ
					gender_net.setInput(age_gender_blob);// gender_net�� �Է���
					Mat gender_res = gender_net.forward();// ���������� ������
					double max1, max2;// ������ ����� �ִ밪�� ������ ����
					Point ptmax1, ptmax2;// ������ ����� �ִ밪�� ��ǥ�� ������ ��ü
					minMaxLoc(gender_res, 0, &max1, 0, &ptmax1);// ������ ���� ����� ptmax1�� ���� �˼�����
					age_net.setInput(age_gender_blob);// age_net�� �Է���
					Mat age_res = age_net.forward();// ���� ������ ������
					minMaxLoc(age_res, 0, &max2, 0, &ptmax2);// ���̿����� ����� ptmax2�� ���� �˼�����
					str1 = format("gender: %s, age: %s", genderList[ptmax1.x].c_str(), ageList[ptmax2.x].c_str());
					int predict_age, predict_gender;// str1���� ������ ������ ���̸� ���ڿ��� ����
					predict_gender = ptmax1.x;// ������ ������ ��������� ����
					predict_age = ptmax2.x;// ���̸� ������ ��������� ����
					if (predict_age == 0 || predict_age == 1)	str2 = "I recommend Milk";// ���������̿� ����
					else if (predict_age == 2 || (predict_gender == 0 && predict_age == 3)) // str2�� ��õ��
						str2 = "I recommend Cider";                                         //��ǰ�� ���ڿ��� 
					else if (predict_gender == 0 && (predict_age == 4 || predict_age == 5)) // �����
						str2 = "I recommend Makgeolli";
					else if (predict_gender == 1 && (predict_age == 3 || predict_age == 4 || predict_age == 5 || predict_age == 6))	 str2 = "I recommend Toreta";
					else if (predict_age == 7 || (predict_gender == 0 && predict_age == 6)) 
						str2 = "I recommend Red ginseng tea";
					ii = 0;
			}
		}
	}
}